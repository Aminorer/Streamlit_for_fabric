import pandas as pd
import numpy as np


def load_sample_data() -> pd.DataFrame:
    """Generate sample dataframe for dashboard pages."""
    rng = pd.date_range(pd.Timestamp.today() - pd.Timedelta(days=29), periods=30)
    categories = ["A", "B", "C"]
    statuses = ["OK", "RUPTURE", "CRITIQUE"]
    data = []
    for cat in categories:
        for date in rng:
            data.append(
                {
                    "category": cat,
                    "date": date,
                    "stock_status": np.random.choice(statuses),
                    "main_rupture_date": date + pd.Timedelta(days=np.random.randint(1, 20)),
                    "criticality_score": np.random.uniform(0, 100),
                    "stock_quantity": np.random.randint(0, 1000),
                }
            )
    return pd.DataFrame(data)
