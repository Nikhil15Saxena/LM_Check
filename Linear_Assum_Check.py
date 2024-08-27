import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import jarque_bera, shapiro
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt

# Function to calculate Variance Inflation Factor (VIF)
def calculate_vif(X):
    try:
        X = X.select_dtypes(include=[np.number])
        if X.isnull().values.any():
            X = X.fillna(0)
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data[vif_data["Variable"] != 'const']
        return vif_data
    except Exception as e:
        st.error(f"An error occurred during VIF calculation: {e}")
        return pd.DataFrame()

# Function to test linear regression assumptions
def test_linear_regression_assumptions(df, target_column, independent_columns):
    X = df[independent_columns]
    y = df[target_column]
    X = sm.add_constant(X)

    # VIF Calculation
    vif_data = calculate_vif(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Breusch-Pagan Test for Homoscedasticity
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_pvalue = bp_test[1]

    # Jarque-Bera Test for Normality
    jb_test = jarque_bera(model.resid)
    jb_pvalue = jb_test[1]

    # Shapiro-Wilk Test for Normality
    sw_test = shapiro(model.resid)
    sw_pvalue = sw_test[1]

    # Durbin-Watson Test for Autocorrelation
    dw_statistic = sm.stats.stattools.durbin_watson(model.resid)

    # Results
    st.header("Statistical Tests Results")
    
    st.write(f"**Breusch-Pagan p-value**: {bp_pvalue:.4f}")
    if bp_pvalue < 0.05:
        st.write("**Homoscedasticity assumption is violated.**")
    else:
        st.write("**Homoscedasticity assumption is satisfied.**")
    
    st.write(f"**Jarque-Bera p-value**: {jb_pvalue:.4f}")
    if jb_pvalue < 0.05:
        st.write("**Normality assumption is violated.**")
    else:
        st.write("**Normality assumption is satisfied.**")
    
    st.write(f"**Shapiro-Wilk p-value**: {sw_pvalue:.4f}")
    if sw_pvalue < 0.05:
        st.write("**Normality assumption is violated.**")
    else:
        st.write("**Normality assumption is satisfied.**")

    st.write(f"**Durbin-Watson statistic**: {dw_statistic:.4f}")
    if dw_statistic < 1.5:
        st.write("**Positive autocorrelation is present (DW < 1.5).**")
    elif dw_statistic > 2.5:
        st.write("**Negative autocorrelation is present (DW > 2.5).**")
    else:
        st.write("**No significant autocorrelation (1.5 < DW < 2.5).**")
    
    st.header("Variance Inflation Factor (VIF)")
    st.write(vif_data)

    # Plotting QQ Plot
    st.header("QQ Plot for Normality Check")
    qqplot_data = qqplot(model.resid, line='s').gca().lines
    plt.title('QQ Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized Residuals')
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Linear Regression Assumptions Checker")
    
    st.header("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        # Show original data shape
        st.write("Original Data Shape:")
        st.write(df.shape)

        # Select target and independent variables
        st.header("Select Variables")
        target_column = st.selectbox("Select the dependent variable (target):", df.columns)
        independent_columns = st.multiselect("Select the independent variables:", df.columns.drop(target_column))
        
        if target_column and independent_columns:
            st.header("Results")
            test_linear_regression_assumptions(df, target_column, independent_columns)

if __name__ == "__main__":
    main()
