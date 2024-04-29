import dash
from dash import html, dcc, Input, Output, dash_table, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from cleaning import create_data, GP_data
from PCA import analyze_depression_clusters

from eda_ml_aky import (load_model, load_scaler, predict_and_evaluate,
                                      create_feature_importance_graph,
                                      create_correlation_heatmap)

from eda_ml_weijiezh import missing_data, load_depression_data, fit_logistic_regression, predict_depression

from catboost_loading import update_catboost
from guassian_process import GP

def load_figure_from_pickle(filename):
    with open(filename, 'rb') as f:
        fig = pickle.load(f)
    return fig

# Example usage:
kmeans_plot = load_figure_from_pickle('figure.pickle')

wei_df = pd.read_csv("WEI_DATA.csv")
gp_data = GP_data()

# missing data plot
missing = missing_data(wei_df)

### TIME SERIES PLOT
mh = wei_df[wei_df.Topic == "Mental Health"]
unique_questions = mh['Question'].unique()

def create_time_series(data, question):
    mean_per_year = data.groupby('YearStart')['DataValue'].mean().reset_index()

    fig = px.line(
        mean_per_year,
        x='YearStart',
        y='DataValue',
        title=f'Time Series for {question}',
        labels={'YearStart': 'Year', 'DataValue': 'Percentage'},
        markers=True
    )
    return fig


'''
HIGH SCHOOL COMPLETION DATA
'''

high_school_completion_data = wei_df[
    wei_df['Question'] == 'High school completion among adults aged 18-24'
]

chloropleth_data = high_school_completion_data[['LocationAbbr', 'DataValue']]
chloropleth_data.dropna(inplace=True)

mean_completion_rates = chloropleth_data.groupby('LocationAbbr')['DataValue'].mean().reset_index()

fig_high_school = px.choropleth(
    mean_completion_rates,
    locations='LocationAbbr',
    locationmode='USA-states',
    color='DataValue',
    color_continuous_scale='Viridis',
    scope='usa',
    title='Average High School Completion Rates among Adults Aged 18-24 by State',
    labels={'DataValue': '% Completion'}
)

fig_high_school.update_layout(
    geo=dict(
        lakecolor='rgb(255, 255, 255)'
    ),
    margin={"r":0,"t":30,"l":0,"b":0}
)


'''
POVERTY DATA
'''

poverty_data = wei_df[
    wei_df['Question'] == 'Living below 150% of the poverty threshold among all people'
]

# Select necessary columns for visualization
violin_data = poverty_data[['Stratification1', 'DataValue']]
violin_data.dropna(inplace=True)  # Remove missing values to ensure clean visualization

# Create a violin plot
fig_violin = px.violin(
    violin_data,
    y='DataValue',
    x='Stratification1',
    color='Stratification1',
    box=True,
    points="all",
    title='Distribution of % Living Below 150% of the Poverty Threshold by Stratification Category',
    height=600
)


'''
UNEMPLOYMENT DATA
'''

unemployment_data = wei_df[
    wei_df['Question'] == 'Unemployment rate among people 16 years and older in the labor force'
]

# Select necessary columns for visualization
unemployment_plot_data = unemployment_data[['Stratification1', 'DataValue']]
unemployment_plot_data.dropna(inplace=True)  # Remove missing values to ensure clean visualization

# Create a violin plot for unemployment rates based on different stratifications
fig_unemployment = px.violin(
    unemployment_plot_data,
    y='DataValue',
    x='Stratification1',
    color='Stratification1',
    box=True,
    points="all",
    title='Distribution of Unemployment Rate by Stratification',
    height=600
)


'''
BUBBLE CHART
'''

# Filter data for the specific question about frequent mental distress among adults
mental_distress_data = wei_df[
    (wei_df['Question'] == 'Frequent mental distress among adults') &
    (wei_df['StratificationCategory1'] == "Race/Ethnicity")
]


# Select necessary columns for visualization
bubble_chart_data = mental_distress_data[['YearStart', 'Stratification1', 'DataValue']]
bubble_chart_data.dropna(inplace=True)  # Remove missing values to ensure clean visualization

# Create a bubble chart
fig_bubble = px.scatter(
    bubble_chart_data,
    x='YearStart',
    y='DataValue',
    color='Stratification1',
    size='DataValue',
    hover_name='Stratification1',
    title='Frequent Mental Distress Among Adults Over Years',
    labels={'YearStart': 'Year', 'DataValue': '% of Frequent Mental Distress'},
    height=600
)

fig_bubble.update_layout(xaxis_title="Year",
                         yaxis_title="Percentage of Frequent Mental Distress",
                         legend_title="Stratification")



# Load and prepare data
df = pd.read_csv('KANE_DATA.csv')
depression_data = load_depression_data("WEI_DATA.csv")
depression_model = fit_logistic_regression(depression_data)

# Load models and scaler
linear_regression_model = load_model('linear_regression_model.joblib')
random_forest_model = load_model('random_forest_model.joblib')
xgboost_model = load_model('gradient_boosting_model.joblib')
scaler = load_scaler('scaler.joblib')

# Prepare test data and metrics for  Kane's model
X_test = df.drop(['Adult Depression','Unnamed: 0', 'YearStart', 'LocationAbbr'], axis=1)
y_test = df['Adult Depression']
_, lr_mse, lr_r2 = predict_and_evaluate(linear_regression_model, X_test, y_test, scaler)
_, rf_mse, rf_r2 = predict_and_evaluate(random_forest_model, X_test, y_test, scaler)
_, xgb_mse, xgb_r2 = predict_and_evaluate(xgboost_model, X_test, y_test, scaler)

# Create heatmap for correlations
heatmap_figure = create_correlation_heatmap(df)

metrics_data = {
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "MSE": [lr_mse, rf_mse, xgb_mse],
    "R^2": [lr_r2, rf_r2, xgb_r2]
}
metrics_df = pd.DataFrame(metrics_data)

importances_df_gb = pd.DataFrame({
    'Feature': X_test.columns.tolist(),
    'Importance': xgboost_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

titles_and_labels = {
    'Adult Binge Prev': ("Binge Drinking and Depression", "Percentage of Adults with Depression", "Binge Drinking Prevalance (%)"),
    'Adult Checkup': ("Regular Medical Visits and Depression", "Percentage of Adults with Depression", "Percentage of Adults with Regular Check-ups"),
    'Adult Obesity': ("Obesity and Depression", "Percentage of Adults with Depression", "Percentage of Adults with Obesity"),
    'Adult Smoking': ("Smoking and Depression", "Percentage of Adults with Depression", "Percentage of Adults who Smoke"),
    'Adults No Ins': ("Insurance and Depression", "Percentage of Adults with Depression", "Percentage of Adults with No Insurance"),
    'Below Poverty Line': ("Poverty and Depression", "Percentage of Adults with Depression", "Percentage Below Poverty Line"),
    'HS Completion 18-24': ("High School Completion and Depression", "Percentage of Adults with Depression", "High School Completion Rate (%)"),
    'Low Fruit Intake Adults': ("Low Fruit Intake and Depression", "Percentage of Adults with Depression", "Low Fruit Intake (%)"),
    'Low Veg Intake Adults': ("Low Vegetable Intake and Depression", "Percentage of Adults with Depression", "Low Vegetable Intake (%)"),
    'No Adult Leisure Activity': ("No Physical Leisure Activity and Depression", "Percentage of Adults with Depression", "No Adult Leisure Activity (%)"),
    'No Broadband': ("Broadband Internet Access and Depression", "Percentage of Adults with Depression", "Percentage Without Broadband"),
    'Per Capita Alcohol': ("Alcohol Consumption and Depression", "Percentage of Adults with Depression", "Per Capita Alcohol Consumption (Liters)"),
    'Unemployment Rate': ("Unemployment and Depression", "Percentage of Adults with Depression", "Unemployment Rate (%)")
}

feature_importance_labels = {
    'Adult Binge Prev': "Binge Drinking",
    'Adult Checkup': "Regular Medical Visits",
    'Adult Obesity': "Obesity",
    'Adult Smoking': "Smoking",
    'Adults No Ins': "Insurance",
    'Below Poverty Line': "Poverty",
    'HS Completion 18-24': "High School Completion",
    'Low Fruit Intake Adults': "Low Fruit Intake",
    'Low Veg Intake Adults': "Low Vegetable Intake",
    'No Adult Leisure Activity': "No Physical Leisure Activity",
    'No Broadband': "Broadband Internet Access",
    'Per Capita Alcohol': "Alcohol Consumption",
    'Unemployment Rate': "Unemployment"
}

importances_df_gb['Feature'] = importances_df_gb['Feature'].replace(feature_importance_labels)


# Set up Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("Predicting Depression from Health Statistics"),
        html.P("This interactive dashboard is designed to explore and analyze health-related data with a focus on understanding how various factors correlate with adult depression. Our goal is to provide a exploratory tool to assess the impact of different health behaviors and social factors, and predict future rates of depression based on historical data."),
        html.H3("Key Features of the Dashboard:"),
        html.Ul([
            html.Li("Data Visualization: Interactive graphs display the relationship between health factors such as smoking, obesity, and binge drinking, and rates of adult depression."),
            html.Li("Model Evaluation: Evaluate the performance of predictive models with metrics like MSE and R²."),
            html.Li("Feature Importance: Visual representations of which factors most significantly impact depression."),
            html.Li("Depression Rate Prediction: Customize predictions for future depression rates using filters for year, location, and demographic groups.")
        ]),
        html.H3("How to Use This Dashboard:"),
        html.P("Navigate through various sections using the menu options. Select different health factors from the dropdown menus to view their correlation with depression rates on the graphs. Use the control panels to adjust parameters and make predictions based on your interests and needs."),
        html.P("This tool is part of an ongoing effort to harness the power of data in combating depression and improving public health outcomes. We encourage you to explore the data, uncover insights, and consider how this information can be used to influence health policies and individual practices."),
    ], className='introduction'),

    html.Div([
        html.H2("Exploratory Data Analysis for the Chronic Disease Indicators Dataset"),
        html.P("This exploratory data analysis (EDA) focuses specifically on the topics social determinants of health and mental health."),
        html.H3("What is the missing mechanism for missing data?"),
        html.P("Consider the following diagram which shows the missing value heatmap:"),
        dcc.Graph(figure = missing),
        html.P("Here we have visualized the missing mechanism for our dataset. We observe that there are columns of completely null values, implying that we may drop them. Furthermore, we can see of the columns that are not completely random, the data appears to be missing by random. Hence, we move forward in this analysis by dropping the columns that are completely null."),

        html.H3("How is the data distributed?"),
        dcc.Graph(figure = kmeans_plot),
        html.P("We are using a data set with multiple columns, in which some of them my be highly correlated if not the same. Hence, we applied a k-means analysis to see the distribution of the data."),
        html.P("What we observe is the clustering features are okay, meaning that the data is not completely random and can be somewhat clustered with little error. In fact, when running a regression on the clusters (with stratification as the dependent variable), the clusters (which now act like features in the regression) actually could explain 94% of the variability within the stratification! (R^2 of 0.937)."),

        html.H3("What state is the most depressed?"),
        dcc.Dropdown(
            id='stratification-dropdown-depression',
            options=[{'label': strat, 'value': strat} for strat in depression_data['Stratification1'].cat.categories],
            value='Overall'  # Default value
        ),
        dcc.Graph(id='depression-graph'),
        html.P("We now begin to ask some questions about the content of the data. As this is focused on mental health, our first natural question would be which state is the most depressed? We have constructed a series of graphs that show the depression rate for each state given a certain stratification. What we noticed in the graphs was the Maine, Tenessee, and West Virginia were usually among the top depressed states. For the least depressed states, we typically saw Hawaii and Nebraska. These seem to line up with the literature and studies very well."),


        html.H3("What state has the highest number of mentally unhealthy days?"),
            dcc.Dropdown(
            id='stratification-dropdown-days',
            options=[{'label': strat, 'value': strat} for strat in depression_data['Stratification1'].cat.categories],
            value='Overall'  # Default value
        ),
        dcc.Graph(id='mental-unhealthy-days-choropleth'),
        html.P("Another statistic that is of interest is the average number of mentally unhealthy days someone experiences within a month. In particular, we once again saw West Virginia and Tenessee, which our least amount of unhealthy days being Nebraska. This reaffirms our conclusion from the last diagram. Another statistic we found suprising was that Asian non-hispanics had the least number of mentally unhealthy days."),




        html.H3("What state has the highest number of routine checkup days within the last year?"),
        dcc.Dropdown(
            id='stratification-dropdown-checkups',
            options=[{'label': strat, 'value': strat} for strat in depression_data['Stratification1'].cat.categories],
            value='Overall'  # Default value
        ),
        dcc.Graph(id='average-checkups-choropleth'),
        html.P("Another statistic that is of interest is the average number of mentally unhealthy days someone experiences within a month. In particular, we once again saw West Virginia and Tenessee, which our least amount of unhealthy days being Nebraska. This reaffirms our conclusion from the last diagram. Another statistic we found suprising was that Asian non-hispanics had the least number of mentally unhealthy days."),




        html.H3("What state had the highest high school completion rates?"),
        dcc.Graph(figure = fig_high_school),
        html.P("In addition to the length between checkups, we also plot the average highschool completion rates. We note that Nebraska, a state that the highest rates of depression, had the lowest graduation rate. Once again, we cannot conclude such causation or correlation, but this statistic still important and interesting to note."),

        html.H3("What demographics have the highest percentage of poverty?"),
        dcc.Graph(figure = fig_violin),
        html.P("We now plot the demographics that have the most severe poverty rates, i.e. being 150% below the poverty threshold. The graph shows the number percentage of a stratification 150% underneath the poverty threshold given that they are already underneath the poverty thershold. Hence, one way to interpret this visualization is the percentage of severely impoverished given that they are impoverished. We note that American Indian and Alaska Native seems to have the overall highest poverty rates, along with Black, Hispanic, and Hawaiian or Pacific Islanders. On the other side of the spectrum, Asian and White had the lowest rates of severe poverty."),

        html.H3("What demographics have the highest percentage of unemployment?"),
        dcc.Graph(figure = fig_unemployment),
        html.P("We now visualize the overall unemployment rates by stratification. Once again, we see that American Indian or Alaska Natives, Black, and Hawaiian or Pacific Islanders have the highest levels of unemployment. On the other hand, once again we see that Asians and White had the lowest rates of unemployments. We cannot conclude any causation from one another, but it is interesting to note such statistic."),

        html.H3("What demographics have the highest amount of mental distress across years? "),
        dcc.Graph(figure = fig_bubble),
        html.P("Consistently American Indian or Alaska Natives have the highest amount of frequent mental distress among the stratification's adults, with Asians having the least amount of frequent mental distress. This may be correlated with unemployment as well as percentage of poverty, as we observe that all three graphs arrive at the same conclusion."),

        html.H3("Time Series for Mental Health Related Topics"),
        *[dcc.Graph(
            id=f'time-series-{question}',
            figure=create_time_series(
                mh[mh['Question'] == question],
                question
            )
        ) for question in unique_questions if question != "Current poor mental health among high school students"],
        html.P("Overall, even when looking at different measures of depression, the trend appears to be an increase. One obvious explanation, given the years we are considering, is that the pandemic exacerbated many mental health struggles. We can see the rates of depression and other mental health issues reach a low during 2020, however spike drastically in the following years. Nonetheless, it is interesting to look at how various social factors and behaviors correlate to depression."),
        html.P("One interesting observation is that the number of Post Partum depressive symptoms among women with a recent live birth seemed to have waned after 2020, i.e. increased at a slower rate. This is interesting to note, as the number of US births significantly decreased after 2020 (the year of the pandemic). Of course further investagtion must be done to determine if there exists a correlation between these two observations, and nothing can be concluded without more research."),

        
    ], className="EDA", style={'width': '60%', 'height': '50%', 'margin': '0 auto'}),


    html.Div([
        html.H2('Relationships Between Various Health Behaviours and Depression'),
        html.P("We now pivot to looking at various health behaviors and their relationships to depression."),
        html.H3("Correlation Heatmap"),
        dcc.Graph(figure=heatmap_figure),
        html.P("Given the vareity of social factors in the data set, it is natural to wonder which correlate or predict depression rates. Below is a correlation heatmap for various social factors and rates of adult depression. These features were chosen based on data completeness and observed correlations."),
        html.P("We note some highly correlated variables. There is a high correlation between no physical activity and smoking, a high correlation bewteen low fruit intake and poverty, and a high correlation between obesity and low fruit intake. On the other hand, there was an unexpected negative correlation between unemployment and binge drinking, a negative correlation bewteen high school completion and poverty, and a negative correlation between having insurance and completing high school.")
    ], className='heatmap-container', style={'width': '60%', 'height': '50%', 'margin': '0 auto'}),



    
    html.Div([
    html.H3("Marginal covariation/correlation between features and depression"),
        # Dropdown component above the Graph component
        dcc.Dropdown(
            id='data-select-dropdown',
            options=[{'label': titles_and_labels[key][0], 'value': key} for key in titles_and_labels],
            value=list(titles_and_labels.keys())[0]  # Default to the first key
        ),
        dcc.Graph(id='data-visualization-graph'),
        html.P("We note some interesting observations about these graphs. Firstly, we observe a strong positive correlation between adults with obesity and depression, smoking and depression, poverty and depression and low fruit intake and depression. However unexpectedly, binge drinking was negatively correlated with depression."),

    ], style={'width': '60%', 'height': '50%', 'margin': '0 auto'}),

    html.Div([
    html.H2("Data Analytics and Machine Learning Predictions"),
    html.Div([
    html.H2("Predictive Analysis Tool"),
    html.P([
        "We now present tools designed to predict and forcast different mental health statistics of individuals likely to suffer from depression based on demographic and geographical factors."
    ]),
    html.P([
        "This predictive tool takes into account various inputs, such as an individual's state of residence and demographic group. By analyzing historical trends and patterns, the tool can forecast depression rates."
    ]),
    html.P([
        "We encourage users to explore different scenarios by selecting their state and demographic group, thereby gaining personalized insights into the prevalence of depression within various communities."
    ])
    ], className="predictive-analysis-tool-introduction"),

    html.H2("Depression Rates Prediction with Ridge Regression"),
        html.P("In this section, we predict the percentage of depression given a stratification, location, and year. We utilize a ridge-regression to counter highly correlated features, along with a transformation of the dependent variable which is percentage of adults that suffer from depression."),
        html.Div([
            html.Label("Select Year:"),
            dcc.Input(id='year-input', type='number', value=2020, style={'marginRight': '10px'}),
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Select Location:"),
            dcc.Dropdown(
                id='location-dropdown',
                options=[{'label': loc, 'value': loc} for loc in depression_data['LocationDesc'].cat.categories],
                value='California',
                style={'marginRight': '10px', 'width': '300px'}
            ),
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Select Demographic:"),
            dcc.Dropdown(
                id='stratification-dropdown',
                options=[{'label': strat, 'value': strat} for strat in depression_data['Stratification1'].cat.categories],
                value='Female',
                style={'width': '300px'}
            ),
        ], style={'marginBottom': '20px'}),
        html.Button('Predict Depression', id='predict-button', n_clicks=0),
        html.Div(id='prediction-result'),






    html.H2("Depression Rates Prediction with Catboost"),
        html.P("In this section, we predict the percentage of depression given a stratification, location, and year. We utilize a catboost, a boosting library that supports categorical features. In particular, we implement a catboost regression."),
        html.Div([
            html.Label("Select Year:"),
            dcc.Input(id='year-input-cat', type='number', value=2020, style={'marginRight': '10px'}),
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Select Location:"),
            dcc.Dropdown(
                id='location-dropdown-cat',
                options=[{'label': loc, 'value': loc} for loc in depression_data['LocationDesc'].cat.categories],
                value='California',
                style={'marginRight': '10px', 'width': '300px'}
            ),
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Select Demographic:"),
            dcc.Dropdown(
                id='stratification-dropdown-cat',
                options=[{'label': strat, 'value': strat} for strat in depression_data['Stratification1'].cat.categories],
                value='Female',
                style={'width': '300px'}
            ),
        ], style={'marginBottom': '20px'}),
        html.Button('Predict Depression', id='predict-button-cat', n_clicks=0),
        html.Div(id='prediction-result-cat'),

        '''html.H2("Average Mentally Unhealthy Days Prediction with Gaussian Processes"),
            html.P("We utilize a Guassian Process to forecast how the number of mentally unhealthy days will evolve in the future. We note that this data was appended with the CDI data from 2001 - 2016. Furthermore, we choose to implement a Gaussian Process for its Bayesian heritage, where the model can learn even under data-scarce situations; this is due to the nature of the dataset, with only 9 observed time steps. Consider using the interactive predictions across stratification."),

        dcc.Dropdown(
            id='strat-dropdown',
            options=[{'label': s, 'value': s} for s in gp_data.columns],
            value='Female',  # Default value
            multi=False,
            clearable=False,
            placeholder="Select Stratification"
        ),
        dcc.Input(
            id='prediction-year',
            type='number',
            value=2023,  # Default year
            min=2023,
            step=1
        ),
        dcc.Graph(id='gp-plot'),  # Plot for the Gaussian Process''',




        html.H2("Feature Analysis and Data Reduction"),
        html.P("In addition our predictions, we've chosen to leverage three different predictive modeling techniques: Linear Regression, Random Forest Regression, and XGBoost. Each model offers unique features that cater to various aspects of our data and can give diverse insights into its underlying patterns."),
        html.P("In particular, we use Linear Regression, XGBoost, and RandomForest to identify feature importance when predicting percentage of adults who are mentally unhealthy."),
], className="model-explanation", style={'width': '60%', 'height': '50%', 'margin': '0 auto'}),


    html.Div([
        html.H2("Feature Importance"),
        dcc.Graph(id='feature-importance-graph', 
              figure=create_feature_importance_graph(importances_df_gb))
    ], style={'width': '60%', 'height': '50%', 'margin': '0 auto'}),
    html.Div([
    html.H3("Analysis of Top Predictors for Depression Rates"),
    html.P([
        "Through feature importance evaluation within our chosen model, key predictors for adult depression rates have emerged. These top factors include 'Low Vegetable Intake', 'Smoking', 'Low Fruit Intake', 'Obesity', and 'Poverty'."
    ]),
    html.P([
        html.Strong("Low Vegetable Intake: "),
        "A diet poor in vegetables is revealed as a significant predictor. Vegetables are abundant in nutrients that are crucial for maintaining balanced brain chemistry and overall health. Low intake may negatively impact mental well-being."
    ]),
    html.P([
        html.Strong("Smoking: "),
        "Tobacco use has long been associated with an array of negative health outcomes, including a higher likelihood of experiencing depressive symptoms. The connection may be due to a combination of physiological, psychological, and social factors."
    ]),
    html.P([
        html.Strong("Low Fruit Intake: "),
        "Similarly to vegetable consumption, fruit intake plays a role in supporting mental health. Fruits are rich in essential vitamins, antioxidants, and fibers, contributing to better health and reduced stress levels."
    ]),
    html.P([
        html.Strong("Obesity: "),
        "Obesity is a complex condition with physical health effects that may compound mental health challenges, including depression. The relationship could be influenced by factors like inflammation, self-image, and societal stigma."
    ]),
    html.P([
        html.Strong("Poverty: "),
        "Economic hardship can profoundly affect mental health, with poverty increasing exposure to a myriad of stressors such as instability, inadequate access to healthcare, and malnutrition, all contributing to higher depression rates."
    ]),
    html.P([
        "Recognizing these predictors highlights the multifaceted nature of depression and suggests that interventions may require a holistic approach, addressing health behaviors, socioeconomic factors, and broad public health policies."
    ])
], className="top-predictors-discussion", style={'width': '60%', 'height': '50%', 'margin': '0 auto'}),

html.Div([
    html.H2("Model Evaluation Metrics"),
    dash_table.DataTable(
        data=metrics_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in metrics_df.columns],
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
], style={'width': '60%', 'height': '50%', 'margin': '0 auto'}),

html.Div([
    html.H3("Interpreting Model Performance:"),
    html.P([
        "Upon thorough evaluation of our models with respect to Mean Squared Error (MSE) and R-squared (R²) metrics, we have found that ",
        html.Strong("XGBoost outperforms"),
        " the other models in both areas. Lower MSE indicates that the XGBoost model has a higher predictive accuracy with smaller errors, while a higher R² value denotes that the model can explain a larger proportion of the variance in the response variable."
    ]),
    html.P([
        "Given this superior performance, we chose to only visualize the feature importance taken from XGBoosting.",
        html.Strong("feature importance within the XGBoost framework"),
        ". Understanding which features most significantly affect the prediction of depression rates helps to interpret the model in the context of our domain knowledge and contributes to stronger and more informed decision making."
    ]),
    html.P("This focus facilitates further insights into the relative impact of each health factor on the likelihood of depression.")
], className="model-performance-explanation", style={'width': '60%', 'height': '50%', 'margin': '0 auto'}),


html.Div([
    html.H2("Conclusion"),
    html.P('This dashboard serves as only a starting point for looking at the causes and trends of depression in the US. From our analysis we come to the following conclusions: '),
    html.Ul([
            html.Li("Rates of depression are increasing."),
            html.Li("Depression has different frequencies depending upon the stratiication and state."),
            html.Li("Many social behaviors such as diet, tobacco use, and economic status offer predictive power for depression."),
            html.Li("Further work and data is needed to provide more accurate long term predictions and unravel the social factors that contribute to mental illness.")
        ]),
    html.P([
        "The data was sourced from the ",
        html.A("Center for Disease Control", href="https://catalog.data.gov/dataset/u-s-chronic-disease-indicators", target="_blank"),
        "."
])   
], className="conclusion", style={'width': '60%', 'height': '50%', 'margin': '0 auto'})

])



### CORRELATION DROPDOWN GRAPHS
@app.callback(
    Output('data-visualization-graph', 'figure'),
    Input('data-select-dropdown', 'value')
)
def update_graph(selected_key):
    x_values = df[selected_key] 
    y_values = df['Adult Depression']

    # Calculate the line of best fit
    slope, intercept = np.polyfit(x_values, y_values, 1)
    line = slope * x_values + intercept

    # Create figure
    fig = go.Figure()

    # Add scatter trace
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Data'))

    # Add line trace
    fig.add_trace(go.Scatter(x=x_values, y=line, mode='lines', name='Fit'))

    # Update plot layout
    fig.update_layout(
        title=f"{titles_and_labels[selected_key][1]} vs. Adult Depression",
        xaxis_title=titles_and_labels[selected_key][2],
        yaxis_title=titles_and_labels[selected_key][1]
    )

    return fig

### DEPRESSIVE STATES DROPDOWN GRAPHS
@app.callback(
    Output('depression-graph', 'figure'),
    [Input('stratification-dropdown-depression', 'value')]
)
def update_graph(selected_stratification):
    stratified_df = wei_df[
        (wei_df['Stratification1'] == selected_stratification) 
        & (wei_df['Question'] == "Depression among adults") &
        (wei_df['YearStart'] == 2022)
    ]

    if stratified_df['DataValue'].max() > 1:
        stratified_df["DataValue"] = stratified_df["DataValue"].div(100)

    stratified_df = stratified_df.groupby('LocationDesc')['DataValue'].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=stratified_df['LocationDesc'], y=stratified_df['DataValue']))

    fig.update_layout(
        title=f'Depression Among Adults by State - {selected_stratification}',
        xaxis_title='State',
        yaxis_title='Depression Rate (%)',
        template='plotly_white',
        height = 1000
    )

    return fig

### AVERAGE MENTALLY UNHEALTHY DAYS CHLOROPLETH
@app.callback(
    Output('mental-unhealthy-days-choropleth', 'figure'),
    [Input('stratification-dropdown-days', 'value')]
)
def update_choropleth(selected_stratification):

    stratified_df = wei_df[
        (wei_df['Stratification1'] == selected_stratification) &
        (wei_df['Question'] == "Average mentally unhealthy days among adults") &
        (wei_df['YearStart'] == 2022)]

    #stratified_df = stratified_df.groupby('LocationAbbr')['DataValue'].max().reset_index()

    fig = px.choropleth(
        stratified_df,
        locations='LocationAbbr',
        locationmode='USA-states',
        color='DataValue',
        color_continuous_scale='Viridis',
        title=f'Average Mentally Unhealthy Days Among Adults by State - {selected_stratification}',
        scope='usa',
        labels={'LocationAbbr': 'State', 'DataValue': 'Avg Mentally Unhealthy Days'},
        template='plotly_white',
        height=600
    )

    return fig

### ROUTINE CHECKUP WITHIN THE PAST YEAR AMONG ADULTS CHLOROPLETH
@app.callback(
    Output('average-checkups-choropleth', 'figure'),
    [Input('stratification-dropdown-checkups', 'value')]
)
def update_choropleth(selected_stratification):

    stratified_df = wei_df[
        (wei_df['Stratification1'] == selected_stratification) &
        (wei_df['Question'] == "Routine checkup within the past year among adults") &
        (wei_df['YearStart'] == 2022)]

    fig = px.choropleth(
        stratified_df,
        locations='LocationAbbr',
        locationmode='USA-states',
        color='DataValue',
        color_continuous_scale='Viridis',
        title=f'Average Days Between Checkups - {selected_stratification}',
        scope='usa',
        labels={'LocationAbbr': 'State', 'DataValue': 'Avg ays between checkups'},
        template='plotly_white',
        height=600
    )

    return fig


@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('year-input', 'value'),
     State('location-dropdown', 'value'),
     State('stratification-dropdown', 'value')]
)
def update_prediction(n_clicks, year, location, stratification):
    if n_clicks:
        probabilities = predict_depression(depression_data, depression_model, year, location, stratification)
        probabilities = probabilities * 100
        return f"Predicted percentage of depression among {stratification} living in {location} in {year} is: {probabilities[0]:.2f}%"
    
@app.callback(
    Output('prediction-result-cat', 'children'),
    [Input('predict-button-cat', 'n_clicks')],
    [State('year-input-cat', 'value'),
     State('location-dropdown-cat', 'value'),
     State('stratification-dropdown-cat', 'value')]
)
def update_prediction(n_clicks, year, location, stratification):
    if n_clicks:
        probabilities = update_catboost(year, location, stratification)
        return f"Predicted percentage of depression among {stratification} living in {location} in {year} is: {probabilities[0]:.2f}%"
    
'''@app.callback(
    Output('gp-plot', 'figure'),
    [Input('strat-dropdown', 'value'),
     Input('prediction-year', 'value')]
)
def update_plot(strat, year):
    # Use the GP function to generate the plot based on selected stratification and year
    return GP(strat, gp_data, year)'''



if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
