# pyMagicStats

---

## English

pyMagicStats is an applied statistics library designed to ensure correctness in statistical and mathematical procedures, offering multiple levels of abstraction. This allows the user to choose the appropriate method for each context. The library is intended for use in reports (Jupyter notebooks, Python scripts, numpy, pandas, matplotlib, seaborn, etc.), data transformation pipelines, and business intelligence tools like PowerBI.

### Features
- **Precision and Correctness:** Implements statistical procedures following best mathematical practices to ensure reliable results.
- **Multiple Levels of Abstraction:** Offers the flexibility to choose the level of detail for analysis, from basic operations to advanced methodologies.
- **Compatibility:** Seamlessly integrates with the Python ecosystem: numpy, pandas, matplotlib, seaborn, and other essential libraries for data analysis and visualization.
- **Commercial-Ready:** Distributed under the Apache License 2.0, which provides extra patent protections, making it suitable for future commercial developments.

### Requirements
- Python 3.6 or higher.
- Main dependencies: numpy, pandas, matplotlib, seaborn, numba.
- Additional dependencies are listed in the `requirements.txt` file.

### Installation
Clone the repository to work directly with the source code:

```bash
git clone https://github.com/your_username/pyMagicStats.git
```

### Usage Examples (Post-Migration)

With the new domain-based architecture, importing modules is more intuitive.

#### 1. Statistical Distributions
Evaluate normality for a set of data:
```python
import numpy as np
from pyMagicStat.distributions.distributions import NormalDistribution

data = np.random.randn(100)
dist = NormalDistribution(data)
results = dist.fit_test()
print(results)
```

#### 2. Statistical Inference (Parametric)
Calculate a confidence interval for the mean:
```python
from pyMagicStat.inference.parametric import PopulationMeanCI

ci = PopulationMeanCI(data, alpha=0.05)
print(ci.calculate_interval())
```

#### 3. Modeling (Linear Regression)
Run a linear regression model:
```python
import pandas as pd
from pyMagicStat.models.regression import RegressionModel

df = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})
model = RegressionModel(data=df, formula='y ~ x')
print(model.summary())
```
