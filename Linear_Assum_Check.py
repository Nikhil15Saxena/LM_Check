import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.stattools as stattools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import jarque_bera, shapiro
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import OLSInfluence

# Function to calculate Variance Inflation Factor
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [stattools.variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data[vif_data["Variable"] != 'const']  # Exclude the constant term
    return vif_data

def durbin_watson_p_value(residuals):
    """
    Compute a p-value for the Durbin-Watson statistic.
    This is a simplified approach based on simulations and assumptions.
    """
    dw_stat = stattools.durbin_watson(residuals)
    num_simulations = 10000
    sim_dws = np.zeros(num_simulations)
    for i in range(num_simulations):
        sim_residuals = np.random.normal(size=len(residuals))
        sim_dws[i] = stattools.durbin_watson(sim_residuals)
    p_value = np.mean(sim_dws <= dw_stat)
    return dw_stat, p_value

def test_linear_regression_assumptions(df, target_column, independent_columns):
    X = df[independent_columns]
    y = df[target_column]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    fittedvalues = model.fittedvalues

    # Statistical Tests
    st.header("Statistical Tests")

    # Breusch-Pagan test for homoscedasticity
    _, p_value, _, _ = het_breuschpagan(residuals, X)
    st.write(f'Breusch-Pagan p-value: {p_value:.4f}')
    homoscedasticity_status = "Homoscedasticity assumption is satisfied." if p_value > 0.05 else "Homoscedasticity assumption is violated."
    st.write(homoscedasticity_status)

    # Jarque-Bera test for normality
    jb_results = jarque_bera(residuals)
    jb_stat, jb_p_value = jb_results
    st.write(f'Jarque-Bera test statistic: {jb_stat:.4f}')
    st.write(f'Jarque-Bera p-value: {jb_p_value:.4f}')
    normality_status_jb = "Normality assumption is satisfied." if jb_p_value > 0.05 else "Normality assumption is violated."
    st.write(normality_status_jb)

    # Shapiro-Wilk test for normality
    sw_stat, sw_p_value = shapiro(residuals)
    st.write(f'Shapiro-Wilk p-value: {sw_p_value:.4f}')
    normality_status_sw = "Normality assumption is satisfied." if sw_p_value > 0.05 else "Normality assumption is violated."
    st.write(normality_status_sw)

    # Durbin-Watson test for autocorrelation
    dw_stat, dw_p_value = durbin_watson_p_value(residuals)
    st.write(f'Durbin-Watson statistic: {dw_stat:.4f}')
    st.write(f'Durbin-Watson p-value (approximate): {dw_p_value:.4f}')
    autocorrelation_status = "Positive autocorrelation is present (DW < 1.5)." if dw_stat < 1.5 else ("Negative autocorrelation is present (DW > 2.5)." if dw_stat > 2.5 else "No significant autocorrelation (1.5 <= DW <= 2.5).")
    st.write(autocorrelation_status)

    # Plots
    st.header("Diagnostic Plots")
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))

    # Linearity
    axs[0, 0].scatter(fittedvalues, residuals)
    axs[0, 0].axhline(y=0, color='r', linestyle='--')
    axs[0, 0].set_xlabel('Fitted Values')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 0].set_title('Linearity Check')

    # Scale-Location Plot
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    axs[0, 1].scatter(fittedvalues, np.sqrt(np.abs(standardized_residuals)))
    axs[0, 1].axhline(y=0, color='r', linestyle='--')
    axs[0, 1].set_xlabel('Fitted Values')
    axs[0, 1].set_ylabel('Sqrt(Abs(Standardized Residuals))')
    axs[0, 1].set_title('Scale-Location Plot')

    # Independence of Residuals
    plot_acf(residuals, lags=40, ax=axs[0, 2])
    axs[0, 2].set_title('Autocorrelation of Residuals')

    # Normality of Residuals
    qqplot(residuals, line='s', ax=axs[1, 0])
    axs[1, 0].set_title('QQ Plot of Residuals')

    # Partial Autocorrelation of Residuals
    plot_pacf(residuals, lags=40, ax=axs[1, 1])
    axs[1, 1].set_title('Partial Autocorrelation of Residuals')

    # Influence Plot
    influence = OLSInfluence(model)
    axs[1, 2].scatter(influence.hat_matrix_diag, influence.cooks_distance[0])
    axs[1, 2].axhline(y=4 / len(residuals), color='r', linestyle='--')
    axs[1, 2].set_xlabel('Leverage')
    axs[1, 2].set_ylabel("Cook's Distance")
    axs[1, 2].set_title("Cook's Distance")

    plt.tight_layout()
    st.pyplot(fig)

    # VIF
    st.header("Variance Inflation Factor (VIF)")
    vif_data = calculate_vif(X)
    st.write(vif_data)

def main():
    st.header("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        st.write("Original Data Shape:")
        st.write(df.shape)

        st.header("Select Dependent and Independent Variables")
        target_column = st.selectbox("Select the dependent variable:", df.columns)
        independent_columns = st.multiselect("Select independent variables:", df.columns[df.columns != target_column])

        if target_column and independent_columns:
            st.header("Filter Data")
            filter_columns = st.multiselect("Select columns to filter:", df.columns)
            filters = {}
            for col in filter_columns:
                unique_values = df[col].unique()
                if pd.api.types.is_numeric_dtype(df[col]):
                    selected_values = st.multiselect(f"Select values for '{col}':", unique_values)
                    filters[col] = selected_values
                else:
                    selected_values = st.multiselect(f"Select values for '{col}':", unique_values)
                    filters[col] = selected_values
            for col, values in filters.items():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df = df[df[col].isin(values)]
                else:
                    df = df[df[col].isin(values)]

            test_linear_regression_assumptions(df, target_column, independent_columns)

if __name__ == "__main__":
    main()
