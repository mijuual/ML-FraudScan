import pytest
import pandas as pd
from scripts/preprocessing import visualize_data
from scripts/preprocessing import map_ip_to_country, add_country_column

def test_visualize_data_runs_without_error():
    # Create dummy dataset with required columns
    data = {
        'class': [0, 1, 0, 1],
        'purchase_value': [10.5, 50.2, 20.1, 30.3],
        'age': [25, 40, 30, 22],
        'source': ['SEO', 'Ads', 'Direct', 'SEO'],
        'browser': ['Chrome', 'Firefox', 'Edge', 'Safari'],
        'sex': ['M', 'F', 'M', 'F']
    }
    df = pd.DataFrame(data)

    # Should run without raising errors
    visualize_data(df)

def test_visualize_data_missing_column():
    # Missing "browser" column
    data = {
        'class': [0, 1],
        'purchase_value': [10, 20],
        'age': [25, 30],
        'source': ['SEO', 'Ads'],
        'sex': ['M', 'F']
    }
    df = pd.DataFrame(data)

    # Expect ValueError
    with pytest.raises(ValueError, match="Missing required column: browser"):
        visualize_data(df)

# --------------------------
# Fixtures (sample data)
# --------------------------
@pytest.fixture
def ip_df():
    return pd.DataFrame({
        'lower_bound_ip_address': [0, 50, 100],
        'upper_bound_ip_address': [49, 99, 150],
        'country': ['USA', 'Canada', 'UK']
    })

@pytest.fixture
def fraud_df():
    return pd.DataFrame({
        'ip_address': [25, 75, 120, 200]
    })


# --------------------------
# Tests for map_ip_to_country
# --------------------------
def test_map_ip_to_country_found(ip_df):
    assert map_ip_to_country(25, ip_df) == 'USA'
    assert map_ip_to_country(75, ip_df) == 'Canada'
    assert map_ip_to_country(120, ip_df) == 'UK'

def test_map_ip_to_country_not_found(ip_df):
    # IP 200 is outside all ranges
    assert map_ip_to_country(200, ip_df) == 'Unknown'


# --------------------------
# Tests for add_country_column
# --------------------------
def test_add_country_column(ip_df, fraud_df):
    result = add_country_column(fraud_df.copy(), ip_df)
    
    # Ensure 'country' column is added
    assert 'country' in result.columns
    
    # Check correct mapping
    expected_countries = ['USA', 'Canada', 'UK', 'Unknown']
    assert list(result['country']) == expected_countries

def test_add_country_column_no_ip_column(ip_df):
    # Missing ip_address column should raise KeyError
    bad_df = pd.DataFrame({'wrong_col': [1, 2, 3]})
    with pytest.raises(KeyError):
        add_country_column(bad_df, ip_df)
