import pandas as pd
from bidict import bidict

def create_data():
    master = pd.read_csv("U.S._Chronic_Disease_Indicators.csv")

    '''
    KANE DATASET CLEANING
    '''

    # Dropping all columns that are full of nans
    kane_data = master.dropna(axis=1, how='all')

    # Filtering for US states
    state_abbreviations = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]
    kane_data = kane_data[kane_data['LocationAbbr'].isin(state_abbreviations)]

    # Filtering to look at overall data as opposed to stratifications
    kane_data = kane_data[kane_data['Stratification1']=='Overall']


    # Filtering for only topics of interest
    topics_to_filter = [
        'Nutrition, Physical Activity, and Weight Status',
        'Social Determinants of Health',
        'Alcohol',
        'Mental Health',
        'Tobacco'
    ]

    kane_data = kane_data[kane_data['Topic'].isin(topics_to_filter)]

    # Only getting needed columns
    kane_data = kane_data.loc[:,['YearStart','LocationAbbr','Question','DataValue','DataValueUnit','DataValueType']]
    kane_data = kane_data[(kane_data['YearStart'] == 2019) | (kane_data['YearStart'] == 2021)]
    kane_data = kane_data.dropna(axis = 0)



    column_name_mapping = bidict({
        'Alcohol use among high school students': 'HS Alcohol Use',
        'Average mentally unhealthy days among adults': 'Adult Mental Unhealth Days',
        'Binge drinking frequency among adults who binge drink': 'Adult Binge Freq',
        'Binge drinking intensity among adults who binge drink': 'Adult Binge Intensity',
        'Binge drinking prevalence among adults': 'Adult Binge Prev',
        'Binge drinking prevalence among high school students': 'HS Binge Prev',
        'Chronic liver disease mortality among all people, underlying cause': 'Liver Disease Mortality',
        'Cigarette smoking during pregnancy among women with a recent live birth': 'Smoking During Pregnancy',
        'Consumed fruit less than one time daily among adults': 'Low Fruit Intake Adults',
        'Consumed fruit less than one time daily among high school students': 'Low Fruit Intake HS',
        'Consumed regular soda at least one time daily among high school students': 'HS Soda Consumption',
        'Consumed vegetables less than one time daily among adults': 'Low Veg Intake Adults',
        'Consumed vegetables less than one time daily among high school students': 'Low Veg Intake HS',
        'Current cigarette smoking among adults': 'Adult Smoking',
        'Current electronic vapor product use among high school students': 'HS Vapor Use',
        'Current smokeless tobacco use among high school students': 'HS Smokeless Tobacco',
        'Current tobacco use of any tobacco product among high school students': 'HS Tobacco Use',
        'Depression among adults': 'Adult Depression',
        'Frequent mental distress among adults': 'Adult Mental Distress',
        'Health insurance coverage after pregnancy among women with a recent live birth': 'Postpartum Ins Coverage',
        'Health insurance coverage in the month before pregnancy among women with a recent live birth': 'Prepartum Ins Coverage',
        'High school completion among adults aged 18-24': 'HS Completion 18-24',
        'Lack of health insurance among adults aged 18-64 years': 'Adults No Ins',
        'Living below 150% of the poverty threshold among all people': 'Below Poverty Line',
        'Met aerobic physical activity guideline among high school students': 'HS Aerobic Activity',
        'No broadband internet subscription among households': 'No Broadband',
        'No leisure-time physical activity among adults': 'No Adult Leisure Activity',
        'Obesity among adults': 'Adult Obesity',
        'Obesity among high school students': 'HS Obesity',
        'Per capita alcohol consumption among people aged 14 years and older': 'Per Capita Alcohol',
        'Postpartum depressive symptoms among women with a recent live birth': 'Postpartum Depression',
        'Proportion of the population protected by a comprehensive smoke-free policy prohibiting smoking in all indoor areas of workplaces and public places, including restaurants and bars': 'Smoke-Free Policy Coverage',
        'Quit attempts in the past year among adult current smokers': 'Adult Quit Attempts',
        'Routine checkup within the past year among adults': 'Adult Checkup',
        'Unemployment rate among people 16 years and older in the labor force': 'Unemployment Rate'
    })

    kane_data['Question'] = kane_data['Question'].apply(lambda x: column_name_mapping.get(x, x))

                                
    kane_data = kane_data[~kane_data['Question'].isin(['Infants who were exclusively breastfed through 6 months',
        'Met aerobic physical activity guideline for substantial health benefits, adults',
                                        'Smoke-Free Policy Coverage',
                                        'Infants who were breastfed at 12 months',
                                        'Current poor mental health among high school students'])]
                        

    values_to_remove=['Adult Binge Freq','HS Alcohol Use','Adult Mental Unhealth Days','HS Binge Prev','Liver Disease Mortality','Smoking During Pregnancy','Low Fruit Intake HS',
                                'HS Soda Consumption','Low Veg Intake HS','HS Vapor Use','HS Smokeless Tobacco','HS Tobacco Use','Adult Mental Distress', 'Postpartum Ins Coverage',
                            'Prepartum Ins Coverage','HS Aerobic Activity','HS Obesity','Postpartum Depression','Adult Quit Attempts','Food insecure in the past 12 months among households',
                    'Adult Binge Intensity']




    kane_data = kane_data[~kane_data['Question'].isin(values_to_remove)]
    kane_data = kane_data[kane_data['DataValueType'].isin(['Crude Prevalence','Per capita alcohol consumption gallons'])]


    # Makibng the questions into the columns
    kane_data = kane_data.pivot_table(index=['YearStart', 'LocationAbbr'], 
                                    columns='Question', 
                                    values='DataValue', 
                                    aggfunc='first')  # Use 'first' to avoid any aggregation conflict
    kane_data.reset_index(inplace=True)
    kane_data = kane_data.dropna()

    kane_data.to_csv('KANE_DATA.csv')

    print("successfully saved KANE_DATA.csv to local directory")

    '''
    WEI + AIDAN DATASET CLEANING
    '''

    wei_data = master[(master.Topic == "Social Determinants of Health") | (master.Topic == "Mental Health")]
    #wei_data = wei_data[["YearStart", "LocationDesc", "DataValue", "LowConfidenceLimit", "HighConfidenceLimit", "StratificationCategory1", "Stratification1"]]

    wei_data['LocationDesc'] = wei_data['LocationDesc'].astype('category')
    wei_data['Stratification1'] = wei_data['Stratification1'].astype('category')

    wei_data.to_csv('WEI_DATA.csv')

    print("successfully saved WEI_DATA.csv to local directory")

def GP_data(file = "U.S._Chronic_Disease_Indicators.csv", aux_file = "USCDI.csv"):

    df = pd.read_csv("U.S._Chronic_Disease_Indicators.csv", low_memory=False)
    df2 = pd.read_csv("USCDI.csv", low_memory=False)

    new_unhealthy_day = df.copy()
    new_unhealthy_day = new_unhealthy_day[["YearStart", "DataValue", "LocationAbbr", "DataSource", "Topic", "Question", "DataValueType", "DataValueUnit", "Stratification1"]]
    new_unhealthy_day = new_unhealthy_day[(new_unhealthy_day.Topic == "Mental Health") & (new_unhealthy_day.Question == 'Average mentally unhealthy days among adults') & (new_unhealthy_day.DataValueType == "Age-adjusted Mean")]

    new_unhealthy_day.dropna(inplace=True)

    early_unhealthy_day = df2[(df2.Topic == "Mental Health") & (df2.Question == 'Recent mentally unhealthy days among adults aged >= 18 years') & (df2.DataValueType == "Age-adjusted Mean")]
    #early_unhealthy_day.dropna(inplace=True)
    early_unhealthy_day = early_unhealthy_day[['YearStart', 'DataValue', 'LocationAbbr', 'DataSource', 'Topic',
        'Question', 'DataValueType', 'DataValueUnit', 'Stratification1']]


    early_unhealthy_day.dropna(inplace=True)
    early_unhealthy_day
    early_unhealthy_day["DataValue"] = pd.to_numeric(early_unhealthy_day["DataValue"])


    master_unhealthy_days = pd.concat([early_unhealthy_day, new_unhealthy_day])
    master_unhealthy_days

    master_unhealthy_days_strat = master_unhealthy_days.groupby(["YearStart", "Stratification1"])["DataValue"].mean().reset_index()
    master_unhealthy_days_strat = master_unhealthy_days_strat.set_index(["YearStart", "Stratification1"]).DataValue.unstack()
    master_unhealthy_days_strat = master_unhealthy_days_strat.dropna(axis = 1)

    return master_unhealthy_days_strat

