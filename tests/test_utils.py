import pandas as pd

from src.utils import add, engineer_iris_features


def test_add():
    assert add(2, 3) == 5.0


def test_engineer_iris_features():
    """Test that engineer_iris_features adds ratio columns correctly."""
    # Create sample data
    df = pd.DataFrame({
        'petal length (cm)': [4.0, 6.0],
        'petal width (cm)': [2.0, 3.0],
        'sepal length (cm)': [5.0, 7.0],
        'sepal width (cm)': [2.5, 3.5]
    })
    
    # Apply feature engineering
    result = engineer_iris_features(df)
    
    # Check that new columns exist
    assert 'petal_ratio' in result.columns
    assert 'sepal_ratio' in result.columns
    
    # Check that ratios are calculated correctly
    assert result['petal_ratio'].iloc[0] == 2.0  # 4.0 / 2.0
    assert result['petal_ratio'].iloc[1] == 2.0  # 6.0 / 3.0
    assert result['sepal_ratio'].iloc[0] == 2.0  # 5.0 / 2.5
    assert result['sepal_ratio'].iloc[1] == 2.0  # 7.0 / 3.5
    
    # Check that original columns are preserved
    assert 'petal length (cm)' in result.columns
    assert 'petal width (cm)' in result.columns
    
    # Check that original dataframe is not modified (copy was made)
    assert 'petal_ratio' not in df.columns
    assert 'sepal_ratio' not in df.columns