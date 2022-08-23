from __future__ import annotations

import json
import logging
import os
import re
import string
from keyword import iskeyword

import numpy as np
import pandas as pd

from pygama.flow.file_db import FileDB
from pygama.lgdo import (
    Array,
    ArrayOfEqualSizedArrays,
    LH5Store,
    Struct,
    Table,
    WaveformTable,
)
from pygama.lgdo.vectorofvectors import build_cl, explode_arrays, explode_cl

log = logging.getLogger(__name__)


class DataLoader:
    """
    Class to facilitate analysis of pygama-processed data across several tiers,
    daq->raw->dsp->hit->evt.  Where possible, we use a SQL-style database of
    cycle files so that a user can quickly select a subset of cycle files for
    interest, and access information at each processing tier.
    Includes methods to build a fileDB, scan available parameter names in each
    file, and available tables (detectors).
    """

    def __init__(
        self,
        config: str | dict = None,
        filedb: str | FileDB = None,
        filedb_config: str | dict = None,
        file_query: str = None,
    ):
        """
        DataLoader init function.

        No hit-level data is loaded in memory at this point.
        User should specify a config file containing DAQ filename format specifiers, etc.

        Parameters
        ----------
        config :
            Configuration file for the DataLoader
        filedb :
            A fileDB must be specified, either with
                1) An instance of FileDB
                2) files written by FileDB.to_disk() (both fileDB and fileDB_config)
                3) config file with enough info for FileDB to perform a DAQ scan
        filedb_config :
            Config file mentioned above for fileDB
        file_query :
            String query that should operate on columns of a fileDB.

        Returns
        -------
        None.
        """
        # declare all member variables
        self.config = None  # dict
        self.fileDB = None  # pygama FileDB
        self.file_list = None  # list
        self.table_list = None
        self.cuts = None
        self.merge_files = False
        self.output_format = "lgdo.Table"
        self.output_columns = None

        # load things if available
        if config is not None:
            if isinstance(config, str):
                with open(config) as f:
                    config = json.load(f)
            self.set_config(config)

        if filedb is None:
            if filedb_config is None:
                raise ValueError("Either filedb or filedb_config is required!")
            else:
                self.fileDB = FileDB(config=filedb_config)
        else:
            if isinstance(filedb, str):
                if filedb_config is None:
                    raise ValueError("Must provide a config file with a fileDB on disk")
                self.fileDB = FileDB(config=filedb_config, file_df=filedb)
            elif isinstance(filedb, FileDB):
                self.fileDB = filedb
            else:
                raise ValueError("filedb must be a string or instance of FileDB ")

        if file_query is not None:
            # set the file_list
            self.file_list = list(self.fileDB.df.query(file_query).index)

    def set_config(self, config: dict):
        """
        Load configiguration file
        """
        self.config = config
        self.data_dir = config["data_dir"]
        self.levels = list(config["levels"].keys())
        self.tiers = {}
        self.cut_priority = {}
        self.evts = {}
        self.tcms = {}
        for level in self.levels:
            self.tiers[level] = config["levels"][level]["tiers"]
            # Set cut priority
            if "parent" in config["levels"][level].keys():  # This level is a TCM
                parent = config["levels"][level]["parent"]
                child = config["levels"][level]["child"]
                self.tcms[level] = config["levels"][level]
                self.cut_priority[level] = self.cut_priority[parent] + 2
                self.cut_priority[child] = self.cut_priority[parent] + 1
                self.evts[child] = {"tcm": level, "parent": parent}

                # Set TCM columns to lookup
                if "tcm_cols" not in config["levels"][level].keys():
                    log.warning(
                        f"TCM levels, e.g. {level}, need to specify the TCM lookup columns"
                    )
            else:
                self.cut_priority[level] = 0

        # Set channel map
        if isinstance(config["channel_map"], dict):
            self.channel_map = config["channel_map"]
        elif isinstance(config["channel_map"], str):
            with open(config["channel_map"]) as f:
                self.channel_map = json.load(f)
        else:
            log.warning("Channel map must be dict or path to JSON file")

    def set_files(self, query: str):
        """
        Set the files of interest, do this before any other operations

        self.file_list is a list of indices corresponding to the row in FileDB

        Parameters
        ----------
        query :
            The file level cuts on the files of interest
            Can be a cut on any of the columns in FileDB
        """
        inds = list(self.fileDB.df.query(query, inplace=False).index)
        if self.file_list is None:
            self.file_list = inds
        else:
            self.file_list += inds

    def get_table_name(self, tier: str, tb: str) -> str:
        """
        Helper function to get the table name for a tier given its table identifier

        Parameters
        ----------
        tier
            Specify the tier whose table format will be used
        tb
            The table identifier that will be passed to the table format
        """
        template = self.fileDB.table_format[tier]
        fm = string.Formatter()
        parse_arr = np.array(list(fm.parse(template)))
        names = list(parse_arr[:, 1])
        if len(names) > 0:
            keyword = names[0]
            args = {keyword: tb}
            table_name = template.format(**args)
        else:
            table_name = template
        return table_name

    def set_datastreams(
        self, ds: list | tuple | np.ndarray, word: str
    ):  # TODO Make this able to handle more complicated requests
        """
        Set self.table_list to the datastreams (detectors) of interest

        Parameters
        -----------
        ds
            Identifies the detectors of interest
            Can be a list of detectorID, serialno, or channels
            or a list of subsystems of interest e.g. "ged"

        word
            The type of identifier used in ds
            Should be a key in the given channel map or a word in the config file
        """
        if self.table_list is None:
            self.table_list = {}

        ds = list(ds)

        found = False
        for level in self.levels:
            tier = self.tiers[level][0]

            template = self.fileDB.table_format[tier]
            fm = string.Formatter()
            parse_arr = np.array(list(fm.parse(template)))
            names = list(parse_arr[:, 1])  # fields required to generate file name
            if len(names) > 0:
                keyword = names[0]

            if word == keyword:
                found = True
                if level in self.table_list.keys():
                    self.table_list[level] += ds
                else:
                    self.table_list[level] = ds

        if not found:
            # look for word in channel map
            raise NotImplementedError

    def set_cuts(self, cuts: dict | list):
        """
        Set the cuts for each level

        Parameters
        ----------
        cut
            The cuts on the columns of the data table, e.g. "trapEftp_cal > 1000"
            If passing a dictionary, the dictionary should be structured as dict[level] = cut
            If passing a list, each item in the array should be able to be applied on one level of tables
            The cuts at different levels will be joined with an "and"
        """
        if self.cuts is None:
            self.cuts = {}
        if isinstance(cuts, dict):
            # verify the correct structure
            for key, value in cuts.items():
                if not (key in self.levels and isinstance(value, str)):
                    raise ValueError(
                        r"cuts dictionary must be in the format \{ level: string \}"
                    )
                if key in self.cuts.keys():
                    self.cuts[key] += " and " + value
                else:
                    self.cuts[key] = value
        elif isinstance(cuts, list):
            raise NotImplementedError
            self.cuts = {}
            # TODO Parse strings to match column names so you don't have to specify which level it is

    def set_output(self, fmt: str, merge_files: bool = None, columns: list = None):
        """
        Set the parameters for the output format of load

        Parameters
        ----------
        fmt
            'lgdo.Table', 'pd.DataFrame', or TBD

        merge_files
            If true, information from multiple files will be merged into one table

        columns
            The columns that should be copied into the output
        """
        self.output_format = fmt
        if merge_files is not None:
            self.merge_files = merge_files
        if columns is not None:
            self.output_columns = columns

    def get_file_list(self) -> pd.DataFrame:
        """
        Returns a copy of FileDB with the dataframe pared down to the current file list
        """
        file_list = self.fileDB.df.iloc[self.file_list]
        return file_list

    def get_filedb(self, columns=None):
        """
        Accessor method
        """
        return self.fileDB

    def get_tiers_for_col(
        self, columns: list | np.ndarray, merge_files: bool = None
    ) -> dict:
        """
        For each column given, get the tiers and tables in that tier where that column can be found

        Parameters
        ----------
        columns
            The columns to look for

        merge_files
            Whether or not to combine the results for all files
            If None, uses self.merge_files

        Returns
        -------
            col_tiers, the tables and tiers that need to be loaded for each column
        """
        col_tiers = {}

        if merge_files is None:
            merge_files = self.merge_files

        if merge_files:
            for file in self.file_list:
                col_inds = set()
                for i, col_list in enumerate(self.fileDB.columns):
                    if not set(col_list).isdisjoint(columns):
                        col_inds.add(i)

                for level in self.levels:
                    for tier in self.tiers[level]:
                        col_tiers[tier] = set()
                        if self.fileDB.df.loc[file, f"{tier}_col_idx"] is not None:
                            for i in range(
                                len(self.fileDB.df.loc[file, f"{tier}_col_idx"])
                            ):
                                if (
                                    self.fileDB.df.loc[file, f"{tier}_col_idx"][i]
                                    in col_inds
                                ):
                                    col_tiers[tier].add(
                                        self.fileDB.df.loc[file, f"{tier}_tables"][i]
                                    )
        else:
            for file in self.file_list:
                col_tiers[file] = {"tables": {}, "columns": {}}
                col_inds = set()
                for i, col_list in enumerate(self.fileDB.columns):
                    if not set(list(col_list)).isdisjoint(columns):
                        col_inds.add(i)

                for level in self.levels:
                    for tier in self.tiers[level]:
                        col_tiers[file]["tables"][tier] = []
                        tier_col_idx = self.fileDB.df.loc[file, f"{tier}_col_idx"]
                        if tier_col_idx is not None:
                            for i in range(len(tier_col_idx)):
                                col_idx = self.fileDB.df.loc[file, f"{tier}_col_idx"][i]
                                if col_idx in col_inds:
                                    col_tiers[file]["tables"][tier].append(
                                        self.fileDB.df.loc[file, f"{tier}_tables"][i]
                                    )
                                    col_in_tier = set.intersection(
                                        set(self.fileDB.columns[col_idx]), set(columns)
                                    )
                                    for c in col_in_tier:
                                        col_tiers[file]["columns"][c] = tier

        return col_tiers

    def gen_entry_list(
        self,
        tcm_level: str = None,
        tcm_table: int | str = None,
        mode: str = "only",
        save_output_columns: bool = False,
        in_memory: bool = True,
        output_file: str = None,
    ) -> pd.DataFrame | None:  # TODO: mode
        """
        Applies cuts to the tables and files of interest
        Does NOT load the column information into memory

        Can only load up to two levels, those joined by tcm_level

        Parameters
        ----------
        tcm_level
            The type of tcm to be used
            If None, will only return information from lowest level

        tcm_table
            The identifier of the table inside this TCM level that you want to use
            If not specified, there must only be one table inside a TCM file in tcm_level

        mode
            'any' : returns every hit in the event if any hit in the event passes the cuts
            'only' : only returns hits that pass the cuts

        save_output_columns
            If true, saves any columns needed for both the cut and the output to the entry_list

        in_memory
            If true, returns the generated entry list in memory

        output_file
            HDF5 file location to write the entry list to disk

        Returns
        -------
        entries
            The entry list containing columns for parent_idx, parent_table, child_idx and output columns if applicable
            Only returned if in_memory is True

        """
        if self.file_list is None:
            raise ValueError("You need to make a query on fileDB, use set_file_list")

        if not in_memory and output_file is None:
            raise ValueError("If in_memory is False, need to specify an output file")

        if in_memory:
            entries = {}

        # Default columns in the entry_list
        if tcm_level is None:
            return self.gen_hit_entries(save_output_columns, in_memory, output_file)
        else:
            # Assumes only one tier in a TCM level
            parent = self.tcms[tcm_level]["parent"]
            child = self.tcms[tcm_level]["child"]
            entry_cols = [
                f"{parent}_table",
                f"{parent}_idx",
                f"{child}_table",
                f"{child}_idx",
            ]
            if save_output_columns:
                for_output = []

            # Find out which columns are needed for any cuts
            cut_cols = {}

            for level in [child, parent]:
                cut_cols[level] = []
                if self.cuts is not None and level in self.cuts.keys():
                    cut = self.cuts[level]
                else:
                    cut = ""
                # String parsing to determine which columns need to be loaded
                terms = re.findall(
                    r"[^\W0-9]\w*", cut
                )  # Matches any valid python variable name
                for term in terms:
                    if term.isidentifier() and not iskeyword(
                        term
                    ):  # Assumes that column names are valid python variable names
                        cut_cols[level].append(term)
                        # Add column to entry_cols if they are needed for both the cut and the output
                        if self.output_columns is not None:
                            if (
                                term in self.output_columns
                                and term not in entry_cols
                                and save_output_columns
                            ):
                                for_output.append(term)
            if save_output_columns:
                entry_cols += for_output
            sto = LH5Store()

            # Make the entry list for each file
            for file in self.file_list:
                # Get the TCM specified
                tcm_tier = self.tiers[tcm_level][
                    0
                ]  # Assumes that each TCM level only has one tier
                tcm_tables = self.fileDB.df.loc[file, f"{tcm_tier}_tables"]
                if len(tcm_tables) > 1 and tcm_table is None:
                    raise ValueError(
                        f"There are {len(tcm_tables)} TCM tables, need to specify which to use"
                    )
                else:
                    if tcm_table is not None:
                        if tcm_table in tcm_tables:
                            tcm_tb = tcm_table
                        else:
                            raise ValueError(
                                f"Table {tcm_table} doesn't exist in {tcm_level}"
                            )
                    else:
                        tcm_tb = tcm_tables[0]
                tcm_path = (
                    self.data_dir
                    + self.fileDB.tier_dirs[tcm_tier]
                    + "/"
                    + self.fileDB.df.iloc[file][f"{tcm_tier}_file"]
                )
                if not os.path.exists(tcm_path):
                    raise FileNotFoundError(f"Can't find TCM file for {tcm_level}")
                tcm_table_name = self.get_table_name(tcm_tier, tcm_tb)
                tcm_lgdo, _ = sto.read_object(tcm_table_name, tcm_path)
                # Have to do some hacky stuff until I get a get_dataframe() method
                tcm_lgdo[self.tcms[tcm_level]["tcm_cols"]["child_idx"]] = Array(
                    nda=explode_cl(tcm_lgdo["cumulative_length"].nda)
                )
                tcm_lgdo.pop("cumulative_length")
                tcm_tb = Table(col_dict=tcm_lgdo)
                f_entries = tcm_tb.get_dataframe()
                renaming = {
                    self.tcms[tcm_level]["tcm_cols"]["child_idx"]: f"{child}_idx",
                    self.tcms[tcm_level]["tcm_cols"]["parent_tb"]: f"{parent}_table",
                    self.tcms[tcm_level]["tcm_cols"]["parent_idx"]: f"{parent}_idx",
                }
                f_entries.rename(columns=renaming, inplace=True)
                # At this point, should have a list of all available hits/evts joined by tcm

                # Perform cuts specified for child or parent level, in that order
                for level in [child, parent]:
                    if level not in self.cuts.keys():
                        continue
                    cut = self.cuts[level]
                    col_tiers = self.get_tiers_for_col(
                        cut_cols[level], merge_files=False
                    )

                    # Tables in first tier of event should be the same for all tiers in one level
                    tables = self.fileDB.df.loc[file, f"{self.tiers[level][0]}_tables"]
                    if self.table_list is not None:
                        if level in self.table_list.keys():
                            tables = self.table_list[level]
                    # Cut any rows of TCM not relating to requested tables
                    f_entries.query(f"{level}_table in {tables}", inplace=True)

                    for tb in tables:
                        tb_table = None
                        tcm_idx = f_entries.query(f"{level}_table == {tb}").index
                        idx_mask = f_entries.loc[tcm_idx, f"{level}_idx"]
                        for tier in self.tiers[level]:
                            tier_path = (
                                self.data_dir
                                + self.fileDB.tier_dirs[tier]
                                + "/"
                                + self.fileDB.df.loc[file, f"{tier}_file"]
                            )
                            if tier in col_tiers[file]["tables"].keys():
                                if tb in col_tiers[file]["tables"][tier]:
                                    table_name = self.get_table_name(tier, tb)
                                    tier_table, _ = sto.read_object(
                                        table_name,
                                        tier_path,
                                        field_mask=cut_cols[level],
                                        idx=idx_mask.tolist(),
                                    )
                                    if tb_table is None:
                                        tb_table = tier_table
                                    else:
                                        tb_table.join(tier_table)
                        if tb_table is None:
                            continue
                        tb_df = tb_table.get_dataframe()
                        tb_df.query(cut, inplace=True)
                        keep_idx = f_entries.query(
                            f"{level}_table == {tb} and {level}_idx in {list(tb_df.index)}"
                        ).index
                        drop_idx = set.symmetric_difference(
                            set(tcm_idx), list(keep_idx)
                        )
                        f_entries.drop(drop_idx, inplace=True)
                        if save_output_columns:
                            for col in tb_df.columns:
                                if col in for_output:
                                    f_entries.loc[keep_idx, col] = tb_df[col].tolist()
                        # end for each table loop
                    # end for each level loop
                f_entries.reset_index(inplace=True, drop=True)
                if in_memory:
                    entries[file] = f_entries
                if output_file:
                    # Convert f_entries DataFrame to Struct
                    f_dict = f_entries.to_dict("list")
                    f_struct = Struct(f_dict)
                    sto.write_object(
                        f_struct, f"entries/{file}", output_file, wo_mode="o"
                    )
                # end for each file loop
        if in_memory:
            return entries

    def gen_hit_entries(
        self,
        save_output_columns: bool = False,
        in_memory: bool = True,
        output_file: str = None,
    ):
        """
        Called by gen_entry_list() to handle the case when tcm_level is None
        Ignores any cuts set on levels above lowest level

        Parameters
        ----------
        save_output_columns
            If true, saves any columns needed for both the cut and the output to the entry_list

        in_memory
            If true, returns the generated entry list in memory

        output_file
            HDF5 file location to write the entry list to disk
        """
        low_level = self.levels[0]
        if in_memory:
            entries = {}
        entry_cols = [f"{low_level}_table", f"{low_level}_idx"]

        # Find out which columns are needed for the cut
        cut_cols = []
        cut = ""
        if self.cuts is not None and low_level in self.cuts.keys():
            cut = self.cuts[low_level]
            # String parsing to determine which columns need to be loaded
            terms = re.findall(
                r"[^\W0-9]\w*", cut
            )  # Matches any valid python variable name
            for term in terms:
                if not iskeyword(term):
                    cut_cols.append(term)
                    # Add column to entry_cols if they are needed for both the cut and the output
                    if self.output_columns is not None:
                        if (
                            term in self.output_columns
                            and term not in entry_cols
                            and save_output_columns
                        ):
                            entry_cols.append(term)
        col_tiers = self.get_tiers_for_col(cut_cols, merge_files=False)
        sto = LH5Store()
        for file in self.file_list:
            f_entries = pd.DataFrame(columns=entry_cols)
            if self.table_list is not None:
                if low_level in self.table_list.keys():
                    tables = self.table_list[low_level]
            else:
                tables = self.fileDB.df.loc[file, f"{self.tiers[low_level][0]}_tables"]

            for tb in tables:
                tb_table = None
                if not cut:
                    tier = self.tiers[low_level][0]
                    tier_path = (
                        self.data_dir
                        + self.fileDB.tier_dirs[tier]
                        + "/"
                        + self.fileDB.df.iloc[file][f"{tier}_file"]
                    )
                    if os.path.exists(tier_path):
                        table_name = self.get_table_name(tier, tb)
                        n_rows = sto.read_n_rows(table_name, tier_path)
                        tb_df = pd.DataFrame(
                            {
                                f"{low_level}_idx": np.arange(n_rows),
                                f"{low_level}_table": tb,
                            }
                        )
                else:
                    for tier in self.tiers[low_level]:
                        if tier in col_tiers[file]["tables"].keys():
                            tier_path = (
                                self.data_dir
                                + self.fileDB.tier_dirs[tier]
                                + "/"
                                + self.fileDB.df.iloc[file][f"{tier}_file"]
                            )
                            if tier in col_tiers[file]["tables"].keys():
                                if tb in col_tiers[file]["tables"][tier]:
                                    table_name = self.get_table_name(tier, tb)
                                    tier_tb, _ = sto.read_object(
                                        table_name, tier_path, field_mask=cut_cols
                                    )
                                    if tb_table is None:
                                        tb_table = tier_tb
                                    else:
                                        tb_table.join(tier_tb)
                    tb_df = tb_table.get_dataframe()
                    tb_df.query(cut, inplace=True)
                    tb_df[f"{low_level}_table"] = tb
                    tb_df[f"{low_level}_idx"] = tb_df.index
                f_entries = pd.concat((f_entries, tb_df), ignore_index=True)[entry_cols]
            if in_memory:
                entries[file] = f_entries
            if output_file:
                # Convert f_entries DataFrame to Struct
                f_dict = f_entries.to_dict("list")
                f_struct = Struct(f_dict)
                sto.write_object(f_struct, f"entries/{file}", output_file, wo_mode="o")
        if in_memory:
            return entries

    def load(
        self,
        entry_list: pd.DataFrame = None,
        in_memory: bool = False,
        output_file: str = None,
        orientation: str = "hit",
        tcm_level: str = None,
    ) -> None | Table | Struct | pd.DataFrame:
        """
        Returns the requested columns in self.output_columns for the entries in the given entry_list

        Parameters
        ----------
        entry_list
            The output of gen_entry_list
            # TODO : support chunked reading of entry_list from disk

        in_memory
            If True, returns the loaded data in memory

        output_file
            If not None, writes the loaded data to the specified file

        orientation
            'hit' or 'evt'
            Specifies the orientation of the output table

        tcm_level
            Which TCM was used to create the entry_list
        """
        if entry_list is None:
            entry_list = self.gen_entry_list(
                tcm_level=tcm_level, save_output_columns=True
            )

        if not in_memory and output_file is None:
            raise ValueError("If in_memory is False, need to specify an output file")

        if orientation == "hit":
            return self.load_hits(entry_list, in_memory, output_file, tcm_level)
        elif orientation == "evt":
            if tcm_level is None:
                if len(self.tcms) == 1:
                    tcm_level = list(self.tcms.keys())[0]
                else:
                    raise ValueError(
                        "Need to specify which coincidence map to use to return event-oriented data"
                    )
            return self.load_evts(entry_list, in_memory, output_file, tcm_level)
        else:
            raise ValueError(
                f"I don't understand what orientation={orientation} means!"
            )

    def load_hits(
        self,
        entry_list: pd.DataFrame,
        in_memory: bool = False,
        output_file: str = None,
        tcm_level: str = None,
    ):
        """
        Called by load() when orientation='hit'
        """
        if tcm_level is None:
            parent = self.levels[0]
            child = None
            load_levels = [parent]
        else:
            parent = self.tcms[tcm_level]["parent"]
            child = self.tcms[tcm_level]["child"]
            load_levels = [parent, child]

        sto = LH5Store()

        if self.merge_files:  # Try to load all information at once
            raise NotImplementedError
        else:  # Not merge_files
            if in_memory:
                load_out = {}
            for file, f_entries in entry_list.items():
                field_mask = []
                # Pre-allocate memory for output columns
                if self.output_columns is None or not self.output_columns:
                    raise ValueError("Need to set output columns to load data")
                for col in self.output_columns:
                    if col not in f_entries.columns:
                        field_mask.append(col)
                col_tiers = self.get_tiers_for_col(field_mask)
                col_dict = f_entries.to_dict("list")
                table_length = len(f_entries)

                # Loop through each table in entry list
                for tb in f_entries[f"{parent}_table"].unique():
                    tcm_idx = f_entries.query(f"{parent}_table == {tb}").index
                    for level in load_levels:
                        idx_mask = f_entries.loc[tcm_idx, f"{level}_idx"].tolist()

                        for tier in self.tiers[level]:
                            if tb in col_tiers[file]["tables"][tier]:
                                tier_path = (
                                    self.data_dir
                                    + self.fileDB.tier_dirs[tier]
                                    + "/"
                                    + self.fileDB.df.iloc[file][f"{tier}_file"]
                                )
                                if os.path.exists(tier_path):
                                    table_name = self.get_table_name(tier, tb)
                                    tier_table, _ = sto.read_object(
                                        table_name,
                                        tier_path,
                                        idx=idx_mask,
                                        field_mask=field_mask,
                                    )
                                    if level == child:
                                        # Explode columns from "evt"-style levels, untested
                                        # Will only work if column has an "nda" attribute
                                        cum_length = build_cl(f_entries[f"{child}_idx"])
                                        exp_cols = explode_arrays(
                                            cum_length,
                                            [a.nda for a in tier_table.values()],
                                        )
                                        tier_table.update(
                                            zip(tier_table.keys(), exp_cols)
                                        )
                                    for col in tier_table.keys():
                                        if isinstance(tier_table[col], Array):
                                            # Allocate memory for column for all channels
                                            if col not in col_dict.keys():
                                                col_dict[col] = np.empty(
                                                    table_length,
                                                    dtype=tier_table[col].dtype,
                                                )
                                            col_dict[col][tcm_idx] = tier_table[col].nda
                                        elif isinstance(tier_table[col], WaveformTable):
                                            wf_table = tier_table[col]
                                            if isinstance(
                                                wf_table["values"],
                                                ArrayOfEqualSizedArrays,
                                            ):
                                                # Allocate memory for columns for all channels
                                                if "wf_dt" not in col_dict.keys():
                                                    col_dict["wf_t0"] = np.empty(
                                                        table_length,
                                                        dtype=wf_table["t0"].dtype,
                                                    )
                                                    col_dict["wf_dt"] = np.empty(
                                                        table_length,
                                                        dtype=wf_table["dt"].dtype,
                                                    )
                                                    col_dict["wf_values"] = np.empty(
                                                        (
                                                            table_length,
                                                            wf_table[
                                                                "values"
                                                            ].nda.shape[1],
                                                        ),
                                                        dtype=wf_table["values"].dtype,
                                                    )
                                                col_dict["wf_t0"][tcm_idx] = wf_table[
                                                    "t0"
                                                ].nda
                                                col_dict["wf_dt"][tcm_idx] = wf_table[
                                                    "dt"
                                                ].nda
                                                col_dict["wf_values"][
                                                    tcm_idx
                                                ] = wf_table["values"].nda
                                            else:  # wf_values is a VectorOfVectors
                                                log.warning(
                                                    f"Not sure how to handle waveforms with values of type {type(tier_table[col]['values'])} yet"
                                                )
                                        else:
                                            log.warning(
                                                f"Not sure how to handle column {col} of type {type(tier_table[col])} yet"
                                            )
                    # end tb loop

                # Convert col_dict to lgdo.Table
                for col in col_dict.keys():
                    nda = np.array(col_dict[col])
                    col_dict[col] = Array(nda=nda)
                f_table = Table(col_dict=col_dict)

                if in_memory:
                    load_out[file] = f_table
                if output_file:
                    sto.write_object(f_table, f"file{file}", output_file, wo_mode="o")
                # end file loop

            if in_memory:
                if self.output_format == "lgdo.Table":
                    return load_out
                elif self.output_format == "pd.DataFrame":
                    for file in load_out.keys():
                        load_out[file] = load_out[file].get_dataframe()
                    return load_out
                else:
                    log.warning(
                        "I don't know how to output "
                        + self.output_format
                        + ", here is a lgdo.Table"
                    )
                    return load_out

    def load_evts(
        self,
        entry_list: pd.DataFrame = None,
        in_memory: bool = False,
        output_file: str = None,
        tcm_level: str = None,
    ):  # TODO
        """
        Called by load() when orientation = 'evt'
        """
        raise NotImplementedError
        parent = self.tcms[tcm_level]["parent"]
        child = self.tcms[tcm_level]["child"]
        load_levels = [parent, child]

        sto = LH5Store()

        if self.merge_files:  # Try to load all information at once
            raise NotImplementedError
        else:  # Not merge_files
            if in_memory:
                load_out = {}
            for file, f_entries in entry_list.items():
                field_mask = []
                f_table = None
                # Pre-allocate memory for output columns
                for col in self.output_columns:
                    if col not in f_entries.columns:
                        f_entries[col] = None
                        field_mask.append(col)
                col_tiers = self.get_tiers_for_col(field_mask)
                col_dict = f_entries.to_dict("list")
                for col in col_dict.keys():
                    nda = np.array(col_dict[col])
                    col_dict[col] = Array(nda=nda)

                # Loop through each table in entry list
                for tb in f_entries[f"{parent}_table"].unique():
                    tcm_idx = f_entries.query(f"{parent}_table == {tb}").index
                    for level in load_levels:
                        idx_mask = f_entries.loc[tcm_idx, f"{level}_idx"]

                        for tier in self.tiers[level]:
                            if tb in col_tiers["tables"][tier]:
                                tier_path = (
                                    self.data_dir
                                    + self.fileDB.tier_dirs[tier]
                                    + "/"
                                    + self.fileDB.df.iloc[file][f"{tier}_file"]
                                )
                                if os.path.exists(tier_path):
                                    table_name = self.get_table_name(tier, tb)
                                    tier_table, _ = sto.read_object(
                                        table_name,
                                        tier_path,
                                        idx=idx_mask,
                                        field_mask=field_mask,
                                    )
                                    for col in tier_table.keys():
                                        f_table[col].nda[tcm_idx] = tier_table[
                                            col
                                        ].tolist()
                    # end tb loop
                if in_memory:
                    load_out[file] = f_table
                if output_file:
                    sto.write_object(f_table, f"file{file}", output_file, wo_mode="o")
                # end file loop

            if in_memory:
                if self.output_format == "lgdo.Table":
                    return load_out
                elif self.output_format == "pd.DataFrame":
                    for file in load_out.keys():
                        load_out[file] = load_out[file].get_dataframe()
                    return load_out
                else:
                    log.warning(
                        "I don't know how to output "
                        + self.output_format
                        + ", here is a lgdo.Table"
                    )
                    return load_out

    def load_detector(self, det_id):  # TODO
        """
        special version of `load` designed to retrieve all file files, tables,
        column names, and potentially calibration/dsp parameters relevant to one
        single detector.
        """
        raise NotImplementedError

    def load_settings(self):  # TODO
        """
        get metadata stored in raw files, usually from a DAQ machine.
        """
        raise NotImplementedError

    def load_dsp_pars(self, query):  # TODO
        """
        access the dsp_pars parameter database (probably JSON format) and do
        some kind of query to retrieve parameters of interest for our file list,
        and return some tables.
        """
        raise NotImplementedError

    def load_cal_pars(self, query):  # TODO
        """
        access the cal_pars parameter database, run a query, and return some tables.
        """
        raise NotImplementedError

    def skim_waveforms(self, mode: str = "hit", hit_list=None, evt_list=None):  # TODO
        """
        handle this one separately because waveforms can easily fill up memory.
        """
        raise NotImplementedError

    def browse(self, query, dsp_config=None):  # TODO
        """
        Interface between DataLoader and WaveformBrowser.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets all fields to their default values, as if this is a newly created DataLoader
        """
        self.file_list = None
        self.table_list = None
        self.cuts = None
        self.merge_files = False
        self.output_format = "lgdo.Table"
        self.output_columns = None


if __name__ == "__main__":
    doc = """
    Demonstrate usage of the `DataLoader` class.
    This could be what we initially run at LNGS - it would try to do the `os.walk`
    method over the existing files, and e.g. scan for existence of various files
    in different stages.  More advanced tests would be moved to a notebook or
    separate script.
    """

    # dl = DataLoader(
    #     config="../../../../loader_config.json",
    #     filedb_config="fileDB_cfg.json",
    #     filedb="fileDB.lh5"
    # )
    # print(dl.get_filedb())

    dl = DataLoader(
        config="../../../../test_loader_config.json",
        filedb_config="../../../../test_fileDB_config.json",
    )
    # print(dl.get_filedb())

    pd.set_option("display.max_colwidth", 10)
    dl.fileDB.get_tables_columns()
    # print(dl.get_filedb().df.columns)

    # dl.fileDB.to_disk("fileDB_cfg.json", "fileDB.lh5")
    # dl2 = DataLoader(config="../../../../loader_config.json", fileDB_config="fileDB_cfg.json", fileDB="fileDB.lh5")
    # dl2.show_fileDB()

    dl.set_files("timestamp == '20220716T104550Z'")
    # print(
    #     dl.get_file_list()[
    #         [
    #             "period",
    #             "run",
    #             "type",
    #             "raw_tables",
    #             "dsp_tables",
    #             "tcm_tables",
    #             "file_status",
    #         ]
    #     ]
    # )

    dl.set_datastreams(np.arange(10), "ch")

    # hit_el = dl.gen_entry_list()

    dl.set_cuts({"hit": "timestamp > 1000"})

    # hit_cut_el = dl.gen_entry_list()

    # tcm_el = dl.gen_entry_list(tcm_level="tcm")

    cols = ["daqenergy", "bl_mean", "channel"]
    dl.set_output(fmt="lgdo.Table", merge_files=False, columns=cols)
    # dl.set_cuts({"hit": "daqenergy > 100"})

    # hit_cut_el = dl.gen_entry_list(save_output_columns=True)

    tcm_el = dl.gen_entry_list(tcm_level="tcm", save_output_columns=True)

    lout = dl.load(tcm_el, in_memory=True, orientation="hit", tcm_level="tcm")
    # print(lout)
