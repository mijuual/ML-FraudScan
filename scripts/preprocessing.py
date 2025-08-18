import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# Visualization function
# --------------------------
def visualize_data(fraud_df):
    """
    Generates exploratory data analysis (EDA) plots for the fraud dataset.
    
    Parameters
    ----------
    fraud_df : pandas.DataFrame
        DataFrame containing the fraud detection dataset. 
        Must include columns: ['class', 'purchase_value', 'age', 'source', 'browser', 'sex'].
    """
    required_cols = ['class', 'purchase_value', 'age', 'source', 'browser', 'sex']
    for col in required_cols:
        if col not in fraud_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 1. Class distribution (fraud vs non-fraud)
    sns.countplot(data=fraud_df, x='class')
    plt.title('Class Distribution (0 = Non-Fraud, 1 = Fraud)')
    plt.close()

    # 2. Purchase Value distribution
    sns.histplot(data=fraud_df, x='purchase_value', bins=30, kde=True)
    plt.title('Distribution of Purchase Value')
    plt.close()

    # 3. Age distribution
    sns.histplot(data=fraud_df, x='age', bins=30, kde=True)
    plt.title('Distribution of Age')
    plt.close()

    # 4. Categorical breakdown: Source, Browser, Sex
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    sns.countplot(data=fraud_df, x='source', ax=axs[0])
    sns.countplot(data=fraud_df, x='browser', ax=axs[1])
    sns.countplot(data=fraud_df, x='sex', ax=axs[2])
    plt.suptitle('Categorical Features')
    plt.close()


# --------------------------
# IP to country mapping
# --------------------------
def map_ip_to_country(user_ip, ip_df):
    """
    Maps a single IP address (as integer) to a country using ip_df ranges.

    Parameters
    ----------
    user_ip : int
        IP address represented as an integer.
    ip_df : pandas.DataFrame
        DataFrame containing ['lower_bound_ip_address', 'upper_bound_ip_address', 'country'].

    Returns
    -------
    str
        The mapped country name, or 'Unknown' if not found.
    """
    match = ip_df[
        (ip_df['lower_bound_ip_address'] <= user_ip) &
        (ip_df['upper_bound_ip_address'] >= user_ip)
    ]
    if not match.empty:
        return match.iloc[0]['country']
    return 'Unknown'


def add_country_column(fraud_df, ip_df):
    """
    Adds a 'country' column to the fraud_df by mapping IP addresses to countries.

    Parameters
    ----------
    fraud_df : pandas.DataFrame
        Must contain column 'ip_address' (integer form).
    ip_df : pandas.DataFrame
        Must contain columns ['lower_bound_ip_address', 'upper_bound_ip_address', 'country'].

    Returns
    -------
    pandas.DataFrame
        fraud_df with an added 'country' column.
    """
    ip_df = ip_df.sort_values(by='lower_bound_ip_address')
    fraud_df['country'] = fraud_df['ip_address'].apply(lambda x: map_ip_to_country(x, ip_df))
    return fraud_df
