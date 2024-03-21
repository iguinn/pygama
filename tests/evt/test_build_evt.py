import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from lgdo import Array, VectorOfVectors, lh5
from lgdo.lh5 import LH5Store

from pygama.evt import build_evt

config_dir = Path(__file__).parent / "configs"
store = LH5Store()


@pytest.fixture(scope="module")
def files_config(lgnd_test_data, tmptestdir):
    tcm_path = "lh5/prod-ref-l200/generated/tier/tcm/phy/p03/r001/l200-p03-r001-phy-20230322T160139Z-tier_tcm.lh5"
    outfile = f"{tmptestdir}/l200-p03-r001-phy-20230322T160139Z-tier_evt.lh5"

    return {
        "tcm": (lgnd_test_data.get_path(tcm_path), "hardware_tcm_1"),
        "dsp": (lgnd_test_data.get_path(tcm_path.replace("tcm", "dsp")), "dsp", "ch{}"),
        "hit": (lgnd_test_data.get_path(tcm_path.replace("tcm", "hit")), "hit", "ch{}"),
        "evt": (outfile, "evt"),
    }


def test_basics(lgnd_test_data, files_config):
    build_evt(
        files_config,
        config=f"{config_dir}/basic-evt-config.json",
        wo_mode="of",
    )

    outfile = files_config["evt"][0]
    f_tcm = files_config["tcm"][0]

    assert "statement" in store.read("/evt/multiplicity", outfile)[0].getattrs().keys()
    assert (
        store.read("/evt/multiplicity", outfile)[0].getattrs()["statement"]
        == "0bb decay is real"
    )
    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 11
    nda = {
        e: store.read(f"/evt/{e}", outfile)[0].view_as("np")
        for e in ["energy", "energy_aux", "energy_sum", "multiplicity"]
    }
    assert (
        nda["energy"][nda["multiplicity"] == 1]
        == nda["energy_aux"][nda["multiplicity"] == 1]
    ).all()
    assert (
        nda["energy"][nda["multiplicity"] == 1]
        == nda["energy_sum"][nda["multiplicity"] == 1]
    ).all()
    assert (
        nda["energy_aux"][nda["multiplicity"] == 1]
        == nda["energy_sum"][nda["multiplicity"] == 1]
    ).all()

    eid = store.read("/evt/energy_id", outfile)[0].view_as("np")
    eidx = store.read("/evt/energy_idx", outfile)[0].view_as("np")
    eidx = eidx[eidx != 999999999999]

    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("np")
    ids = ids[eidx]
    assert ak.all(ids == eid[eid != 0])


def test_spms_module(lgnd_test_data, files_config):
    build_evt(
        files_config,
        config=f"{config_dir}/spms-module-config.json",
        wo_mode="of",
    )

    outfile = files_config["evt"][0]

    evt = lh5.read("/evt", outfile)

    mask = evt._pulse_mask
    assert isinstance(mask, VectorOfVectors)
    assert len(mask) == 10
    assert mask.ndim == 3

    full = evt.spms_amp_full.view_as("ak")
    amp = evt.spms_amp.view_as("ak")

    assert ak.all(full[mask.view_as("ak")] == amp)

    wo_empty = evt.spms_amp_wo_empty.view_as("ak")
    assert ak.all(wo_empty == amp[ak.count(amp, axis=-1) > 0])

    rawids = evt.rawid.view_as("ak")
    assert rawids.ndim == 2
    assert ak.count(rawids) == 30

    rawids_wo_empty = evt.rawid_wo_empty.view_as("ak")
    assert ak.count(rawids_wo_empty) == 7


def test_vov(lgnd_test_data, files_config):
    build_evt(
        files_config,
        config=f"{config_dir}/vov-test-evt-config.json",
        wo_mode="of",
    )

    outfile = files_config["evt"][0]
    f_tcm = files_config["tcm"][0]

    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 12
    vov_ene, _ = store.read("/evt/energy", outfile)
    vov_aoe, _ = store.read("/evt/aoe", outfile)
    arr_ac, _ = store.read("/evt/multiplicity", outfile)
    vov_aoeene, _ = store.read("/evt/energy_times_aoe", outfile)
    vov_eneac, _ = store.read("/evt/energy_times_multiplicity", outfile)
    arr_ac2, _ = store.read("/evt/multiplicity_squared", outfile)
    assert isinstance(vov_ene, VectorOfVectors)
    assert isinstance(vov_aoe, VectorOfVectors)
    assert isinstance(arr_ac, Array)
    assert isinstance(vov_aoeene, VectorOfVectors)
    assert isinstance(vov_eneac, VectorOfVectors)
    assert isinstance(arr_ac2, Array)
    assert (np.diff(vov_ene.cumulative_length.nda, prepend=[0]) == arr_ac.nda).all()

    vov_eid = store.read("/evt/energy_id", outfile)[0].view_as("ak")
    vov_eidx = store.read("/evt/energy_idx", outfile)[0].view_as("ak")
    vov_aoe_idx = store.read("/evt/aoe_idx", outfile)[0].view_as("ak")

    ids = store.read("hardware_tcm_1/array_id", f_tcm)[0].view_as("ak")
    ids = ak.unflatten(ids[ak.flatten(vov_eidx)], ak.count(vov_eidx, axis=-1))
    assert ak.all(ids == vov_eid)

    arr_ene = store.read("/evt/energy_sum", outfile)[0].view_as("ak")
    assert ak.all(arr_ene == ak.nansum(vov_ene.view_as("ak"), axis=-1))
    assert ak.all(vov_aoe.view_as("ak") == vov_aoe_idx)


def test_graceful_crashing(lgnd_test_data, files_config):
    with pytest.raises(TypeError):
        build_evt(files_config, None, wo_mode="of")

    conf = {"operations": {}}
    with pytest.raises(ValueError):
        build_evt(files_config, conf, wo_mode="of")

    conf = {"channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]}}
    with pytest.raises(ValueError):
        build_evt(files_config, conf, wo_mode="of")

    conf = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "outputs": ["foo"],
        "operations": {
            "foo": {
                "channels": "geds_on",
                "aggregation_mode": "banana",
                "expression": "hit.cuspEmax_ctc_cal > a",
                "parameters": {"a": 25},
                "initial": 0,
            }
        },
    }
    with pytest.raises(ValueError):
        build_evt(
            files_config,
            conf,
            wo_mode="of",
        )


def test_query(lgnd_test_data, files_config):
    build_evt(
        files_config,
        config=f"{config_dir}/query-test-evt-config.json",
        wo_mode="of",
    )
    outfile = files_config["evt"][0]

    assert len(lh5.ls(outfile, "/evt/")) == 12


def test_vector_sort(lgnd_test_data, files_config):
    conf = {
        "channels": {"geds_on": ["ch1084803", "ch1084804", "ch1121600"]},
        "outputs": ["acend_id", "t0_acend", "decend_id", "t0_decend"],
        "operations": {
            "acend_id": {
                "channels": "geds_on",
                "aggregation_mode": "gather",
                "query": "hit.cuspEmax_ctc_cal>25",
                "expression": "tcm.array_id",
                "sort": "ascend_by:dsp.tp_0_est",
            },
            "t0_acend": {
                "aggregation_mode": "keep_at_ch:evt.acend_id",
                "expression": "dsp.tp_0_est",
            },
            "decend_id": {
                "channels": "geds_on",
                "aggregation_mode": "gather",
                "query": "hit.cuspEmax_ctc_cal>25",
                "expression": "tcm.array_id",
                "sort": "descend_by:dsp.tp_0_est",
            },
            "t0_decend": {
                "aggregation_mode": "keep_at_ch:evt.acend_id",
                "expression": "dsp.tp_0_est",
            },
        },
    }

    build_evt(
        files_config,
        conf,
        wo_mode="of",
    )

    outfile = files_config["evt"][0]

    assert os.path.exists(outfile)
    assert len(lh5.ls(outfile, "/evt/")) == 4
    vov_t0, _ = store.read("/evt/t0_acend", outfile)
    nda_t0 = vov_t0.to_aoesa().view_as("np")
    assert ((np.diff(nda_t0) >= 0) | (np.isnan(np.diff(nda_t0)))).all()
    vov_t0, _ = store.read("/evt/t0_decend", outfile)
    nda_t0 = vov_t0.to_aoesa().view_as("np")
    assert ((np.diff(nda_t0) <= 0) | (np.isnan(np.diff(nda_t0)))).all()


def test_chname_fmt(lgnd_test_data, files_config):
    f_config = f"{config_dir}/basic-evt-config.json"

    with pytest.raises(ValueError):
        build_evt(files_config, f_config, wo_mode="of", chname_fmt="ch{{}}")
    with pytest.raises(NotImplementedError):
        build_evt(files_config, f_config, wo_mode="of", chname_fmt="ch{tcm_id}")
    with pytest.raises(ValueError):
        build_evt(files_config, f_config, wo_mode="of", chname_fmt="apple{}banana")
