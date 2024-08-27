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
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Function to calculate Durbin-Watson p-value
def durbin_watson_p_value(residuals):
    dw_stat = stattools.durbin_watson(residuals)
    num_simulations = 10000
    sim_dws = np.zeros(num_simulations)
    for i in range(num_simulations):
        sim_residuals = np.random.normal(size=len(residuals))
        sim_dws[i] = stattools.durbin_watson(sim_residuals)
    p_value = np.mean(sim_dws <= dw_stat)
    return dw_stat, p_value

# Function to calculate Variance Inflation Factor (VIF)
def calculate_vif(X):
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [stattools.variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Streamlit app
st.title("Linear Regression Assumptions Checker")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Allow user to select dependent and independent variables
    st.sidebar.header("Variable Selection")
    target_column = st.sidebar.selectbox("Select dependent variable", df.columns)
    independent_vars = st.sidebar.multiselect("Select independent variables", df.columns)
    
    if target_column and independent_vars:
        # Display selected variables
        st.write(f"Dependent Variable: {target_column}")
        st.write(f"Independent Variables: {independent_vars}")

        # Prepare data for modeling
        X = df[independent_vars]
        y = df[target_column]
        X = sm.add_constant(X)

        # Fit the model
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        fittedvalues = model.fittedvalues

        # Linearity
        st.subheader("Linearity Check")
        fig, ax = plt.subplots()
        ax.scatter(fittedvalues, residuals)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Linearity Check')
        st.pyplot(fig)

        # Scale-Location Plot
        st.subheader("Scale-Location Plot")
        standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
        fig, ax = plt.subplots()
        ax.scatter(fittedvalues, np.sqrt(np.abs(standardized_residuals)))
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Sqrt(Abs(Standardized Residuals))')
        ax.set_title('Scale-Location Plot')
        st.pyplot(fig)

        # Autocorrelation of Residuals
        st.subheader("Autocorrelation of Residuals")
        fig, ax = plt.subplots()
        plot_acf(residuals, lags=40, ax=ax)
        ax.set_title('Autocorrelation of Residuals')
        st.pyplot(fig)

        # Normality of Residuals
        st.subheader("QQ Plot of Residuals")
        fig, ax = plt.subplots()
        qqplot(residuals, line='s', ax=ax)
        ax.set_title('QQ Plot of Residuals')
        st.pyplot(fig)

        # Partial Autocorrelation of Residuals
        st.subheader("Partial Autocorrelation of Residuals")
        fig, ax = plt.subplots()
        plot_pacf(residuals, lags=40, ax=ax)
        ax.set_title('Partial Autocorrelation of Residuals')
        st.pyplot(fig)

        # Influence Plot
        st.subheader("Cook's Distance")
        influence = OLSInfluence(model)
        fig, ax = plt.subplots()
        ax.scatter(influence.hat_matrix_diag, influence.cooks_distance[0])
        ax.axhline(y=4 / len(residuals), color='r', linestyle='--')
        ax.set_xlabel('Leverage')
        ax.set_ylabel("Cook's Distance")
        ax.set_title("Cook's Distance")
        st.pyplot(fig)

        # Leverage, Standardized Residuals, and Cook's Distance
        st.subheader("Leverage vs Residuals")
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].scatter(model.get_influence().hat_matrix_diag, residuals)
        axs[0].axhline(y=0, color='r', linestyle='--')
        axs[0].set_xlabel('Leverage')
        axs[0].set_ylabel('Residuals')
        axs[0].set_title('Leverage vs Residuals')

        axs[1].scatter(model.get_influence().hat_matrix_diag, model.get_influence().resid_studentized_internal)
        axs[1].axhline(y=0, color='r', linestyle='--')
        axs[1].set_xlabel('Leverage')
        axs[1].set_ylabel('Standardized Residuals')
        axs[1].set_title('Leverage vs Standardized Residuals')

        st.pyplot(fig)

        # Statistical Tests
        st.subheader("Statistical Tests")

        # Breusch-Pagan test for homoscedasticity
        _, p_value, _, _ = het_breuschpagan(residuals, X)
        st.write(f'Breusch-Pagan p-value: {p_value:.4f}')
        if p_value > 0.05:
            st.write("Homoscedasticity assumption is satisfied.")
        else:
            st.write("Homoscedasticity assumption is violated.")

        # Jarque-Bera test for normality
        jb_results = jarque_bera(residuals)
        jb_stat, jb_p_value = jb_results[0], jb_results[1]
        st.write(f'Jarque-Bera test statistic: {jb_stat:.4f}')
        st.write(f'Jarque-Bera p-value: {jb_p_value:.4f}')
        if jb_p_value > 0.05:
            st.write("Normality assumption is satisfied.")
        else:
            st.write("Normality assumption is violated.")

        # Shapiro-Wilk test for normality
        sw_stat, sw_p_value = shapiro(residuals)
        st.write(f'Shapiro-Wilk p-value: {sw_p_value:.4f}')
        if sw_p_value > 0.05:
            st.write("Normality assumption is satisfied.")
        else:
            st.write("Normality assumption is violated.")

        # Durbin-Watson test for autocorrelation
        dw_stat, dw_p_value = durbin_watson_p_value(residuals)
        st.write(f'Durbin-Watson statistic: {dw_stat:.4f}')
        st.write(f'Durbin-Watson p-value (approximate): {dw_p_value:.4f}')
        if dw_stat < 1.5:
            st.write("Positive autocorrelation is present (DW < 1.5).")
        elif dw_stat > 2.5:
            st.write("Negative autocorrelation is present (DW > 2.5).")
        else:
            st.write("No significant autocorrelation (1.5 <= DW <= 2.5).")

        # Variance Inflation Factor (VIF)
        vif_data = calculate_vif(X)
        vif_data = vif_data[vif_data["Variable"] != "const"]
        st.subheader("Variance Inflation Factor (VIF)")
        st.write(vif_data)

        # Test for additional assumption of autocorrelation
        st.subheader("Additional Tests")

        # Add any additional tests or information here

        st.write("All assumptions tests are completed. Please review the results above.")

    else:
        st.write("Please select both dependent and independent variables.")
