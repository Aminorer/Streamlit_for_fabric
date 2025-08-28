import pandas as pd
import db_utils


def test_discover_prediction_weeks(monkeypatch):
    monkeypatch.setattr(
        db_utils,
        "find_pred_tables",
        lambda: [
            "pred_amz_man_20240102",
            "pred_amz_dis_20240102",
            "pred_amz_man_20240109",
            "pred_ebay_man_20240102",
        ],
    )
    weeks = db_utils.discover_prediction_weeks("amz")
    assert weeks == [
        "02/01/2024 - Semaine 1",
        "09/01/2024 - Semaine 2",
    ]


def test_load_multi_week_predictions(monkeypatch):
    monkeypatch.setattr(
        db_utils,
        "find_pred_tables",
        lambda: [
            "pred_amz_man_20240102",
            "pred_amz_man_20240109",
            "pred_amz_dis_20240102",
        ],
    )
    calls = []

    def fake_load_prediction_data(
        table_name,
        *,
        brands=None,
        seasons=None,
        sizes=None,
        start_date=None,
        end_date=None,
    ):
        calls.append(
            {
                "table_name": table_name,
                "brands": brands,
                "seasons": seasons,
                "sizes": sizes,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        return pd.DataFrame({"table": [table_name]})

    monkeypatch.setattr(db_utils, "load_prediction_data", fake_load_prediction_data)

    filters = {
        "brands": ["MICHELIN"],
        "seasons": ["ETE"],
        "sizes": ["205/55R16"],
        "start_date": None,
        "end_date": None,
    }
    selected = [
        "02/01/2024 - Semaine 1",
        "09/01/2024 - Semaine 2",
    ]
    result = db_utils.load_multi_week_predictions("amz", "man", selected, filters)

    assert set(result.keys()) == set(selected)
    assert result[selected[0]].iloc[0]["table"] == "pred_amz_man_20240102"
    assert result[selected[1]].iloc[0]["table"] == "pred_amz_man_20240109"
    assert len(calls) == 2
    assert calls[0]["brands"] == ["MICHELIN"]
    assert calls[0]["seasons"] == ["ETE"]
    assert calls[0]["sizes"] == ["205/55R16"]
    assert calls[0]["start_date"] is None
    assert calls[0]["end_date"] is None
    assert calls[1]["table_name"] == "pred_amz_man_20240109"


def test_load_multi_week_predictions_default_under_four(monkeypatch):
    monkeypatch.setattr(
        db_utils,
        "find_pred_tables",
        lambda: [
            "pred_amz_man_20240102",
            "pred_amz_man_20240109",
            "pred_amz_man_20240116",
        ],
    )

    def fake_load_prediction_data(table_name, **kwargs):
        return pd.DataFrame({"table": [table_name]})

    monkeypatch.setattr(db_utils, "load_prediction_data", fake_load_prediction_data)

    result = db_utils.load_multi_week_predictions("amz", "man", filters={})

    expected_keys = [
        "02/01/2024 - Semaine 1",
        "09/01/2024 - Semaine 2",
        "16/01/2024 - Semaine 3",
    ]
    assert list(result.keys()) == expected_keys
