import importlib.util
from pathlib import Path
import locale
import pandas as pd

# Dynamically load the module since the filename starts with a digit
spec = importlib.util.spec_from_file_location(
    "analyse_module", Path("pages") / "1_Analyse_Comparative.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

plot_accuracy_evolution = module.plot_accuracy_evolution


def test_plot_accuracy_evolution_chronological(monkeypatch):
    # Ensure a non-French locale is active
    previous_locale = locale.setlocale(locale.LC_TIME)
    locale.setlocale(locale.LC_TIME, "C")
    try:
        acc_df = pd.DataFrame(
            {
                "week": [
                    "15/01/2024 - Semaine 3",
                    "01/01/2024 - Semaine 1",
                    "08/01/2024 - Semaine 2",
                ],
                "accuracy": [0.9, 0.8, 0.85],
            }
        )
        captured = {}

        def fake_plotly_chart(fig, use_container_width=True):
            captured["fig"] = fig

        monkeypatch.setattr(module.st, "plotly_chart", fake_plotly_chart)
        plot_accuracy_evolution(acc_df)
    finally:
        locale.setlocale(locale.LC_TIME, previous_locale)

    x_vals = list(captured["fig"].data[0].x)
    assert x_vals == [
        "01/01/2024 - Semaine 1",
        "08/01/2024 - Semaine 2",
        "15/01/2024 - Semaine 3",
    ]
