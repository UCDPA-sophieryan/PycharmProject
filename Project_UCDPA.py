import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import folium

#Import CSV file into DataFrame
world_rankings = pd.read_csv('World_University_Ranks_2020.csv')
print(world_rankings.head())

#change column data type
world_rankings['Percentage_Female'] = pd.to_numeric(world_rankings['Percentage_Female'] , errors='coerce')

# Checking of missing data
world_rankings.isnull().sum()
#Dropping duplicates
world_rankings.drop_duplicates(subset=['University'])
world_rankings.sort_values('Rank_Char', ascending=False)

top_20 = world_rankings.iloc[0:21]
print(top_20.head())

#creating dictionary

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = {'University_Name': ['Trinity College Dublin', 'Royal College of Surgeons in Ireland (RCSI)','University College Dublin',
             'National University of Ireland, Galway', 'Maynooth University', 'University College Cork', 'University of Limerick',
        'Dublin City University', 'Technological University Dublin'], 'Global_Ranking': [114, 142,158,188,199,208,288,302,401]}
irish_unis = pd.DataFrame(df)
print(irish_unis)

sns.set_theme(style="darkgrid")
world_rankings = pd.read_csv('World_University_Ranks_2020.csv')
irish_unis = pd.DataFrame(df)
fig, ax = plt.subplots()
sns.scatterplot(data=irish_unis, x="Global_Ranking", y="University_Name", marker='o')
ax.set_title('How Irish universities feature in global rankings')
ax.set(xlabel="Global ranking score", ylabel='Irish University Name')
ax.set_yticklabels = 'Amount of International Students'
plt.show()

#functions
def ireland_universities(str):
    print(str)
    return;

#finding where Ireland uni's feature
ireland_features = world_rankings[world_rankings["Country"].str.contains("Ireland")]
print(ireland_features)

ireland_universities('Ireland has multiple universities on a global ranking scale')

def top_ten_global_unis(str):
    print(str)
    return;

#creating a list
irish_unis_list = ['Trinity College Dublin', 'Royal College of Surgeons in Ireland (RCSI)','University College Dublin',
             'National University of Ireland, Galway', 'Maynooth University', 'University College Cork', 'University of Limerick',
        'Dublin City University', 'Technological University Dublin']
print(irish_unis_list[0])
print(irish_unis_list[2])
print(len(irish_unis_list))



#for loop to use in project
count = 0
for ireland in world_rankings['Country']:
        if (ireland == 'Ireland'):
            count += 1
    print(count, 'instances of an Irish university found in top 200 rankings')

#finding where Ireland uni's feature
ireland_features = world_rankings[world_rankings["Country"].str.contains("Ireland")]
for i in ireland_features:
    print(i)

#iterrows
import pandas as pd
world_rankings = pd.read_csv('World_University_Ranks_2020.csv')
for index, row in world_rankings.head(n=2).iterrows():
     print(index, row)

#numpy
import numpy as np
import pandas as pd
world_rankings = pd.read_csv('World_University_Ranks_2020.csv')
arr = np.array([world_rankings['Country']])
print(arr.dtype)
print(arr.shape)

#first visualisation
import seaborn as sns
world_rankings = pd.read_csv('World_University_Ranks_2020.csv')
df = pd.DataFrame(world_rankings.loc[0:19])
fig, ax = plt.subplots()
sns.countplot(x='Country', data=df).set_title('Top 20 Global University Overview')
ax.set(xlabel="Top 20 Countries", ylabel='Number of universities per country')
plt.show()
fig.savefig('Insight1.jpg')
#insight - America has the highest amount of top ranking universities in the Top 20 global rankings

#second visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme(style="whitegrid")
top_10 =  world_rankings.loc[0:9]
fig, ax = plt.subplots()
g = sns.barplot(y='University', x='International_Students_Percent', data=top_10, orient='h')
ax.set(xlabel="% of international students", ylabel = "Top 10 Global Universities")
g.set_title('Top 10 International Student Overview')
plt.show()
fig.savefig('Insight2.jpg')
#insight - analysis is that Imperial college london has highest amount of international students in top 10 unis
def top_ten_global_unis(str):
    print(str)
    return;

top_ten_global_unis('Imperial college london has highest amount of international students in top 10 universities')

#third visualisation
sns.set_theme(style="darkgrid")
top_20 = world_rankings.loc[0:19]
top_20_sorted = top_20.sort_values('Students_Percentage_Female',ascending=False)
g = sns.catplot(data=top_20_sorted,kind="bar",x="Students_Percentage_Female",
                y="Number_students",hue="Country",ci="sd", palette="dark",
                alpha=.6,height=6)
g.despine(left=True)
g.set_axis_labels("% Female Students","Total Number Students (00s)")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Top 20 Universities Female Makeup")
plt.savefig('Insight3.jpg')
plt.show()
#insight - analysis is that Canada has highest percentage of female students in a university top 20 unis globally

#fourth visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="darkgrid")
world_rankings = pd.read_csv('World_University_Ranks_2020.csv')
ireland_features = world_rankings[world_rankings["Country"].str.contains("Ireland")]
fig, ax = plt.subplots()
sns.scatterplot(data=ireland_features, x="Score_Rank", y="University", marker='o')
ax.set_title('How Irish universities feature in global rankings')
ax.set(xlabel="Global ranking score", ylabel='Irish University Name')
ax.set_yticklabels = 'Amount of International Students'
fig.savefig('Insight4.jpg')
plt.show()

ireland_universities('Ireland has 5 universities in top 200 global rankings')
#insight - Ireland has 5 universities in top 200 global rankings, and we can see who are the top three universities within Ireland

#fifth visualisation
import seaborn as sns
import matplotlib.pyplot as plt
world_rankings = pd.read_csv('World_University_Ranks_2020.csv')
df = pd.DataFrame(world_rankings.loc[0:19])
fig, ax = plt.subplots()
g = sns.barplot(x="Citations", y="Country",
           hue='Country',
            data=df)
ax.set_title('Citation chart for top countries')
ax.set(xlabel="Citation rate", ylabel='Total per country')

plt.legend(loc='upper right')
plt.show()
fig.savefig('Insight5.jpg')
# insight - united states has the most citations in top 20 unis



#Bokeh plot
from bokeh.models import ColumnDataSource, RadioGroup
from bokeh.plotting import figure, output_file, show
output_file("hbar_stack.html")
source = ColumnDataSource(data=dict(
    top_10_uni_rank= [1,2,3,4,5,6,7,8,9,10],
    top_10_female_students= [46,34,47,43,39,45,49,50,46,38],
    top_10_male_students= [54,66,53,57,61,55,51,50,54,62],
))
p = figure(plot_width=400, plot_height=400, title="Student gender breakdown of top 10 universities",
           x_axis_label='% of student gender breakdown', y_axis_label='Top 10 Universities')
p.hbar_stack(['top_10_female_students', 'top_10_male_students'], y='top_10_uni_rank',
             height=0.8, color=("red", "blue"), source=source, legend_label=('Female', 'Male'))
p.legend.orientation = "horizontal"
p.legend.location = "bottom_right"
show(p)

#Merging dataframes

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
irish_unis_dict = {'Trinity College Dublin': 114, 'Royal College of Surgeons in Ireland (RCSI)':142,'University College Dublin':158,
             'National University of Ireland, Galway':188, 'Maynooth University':199, 'University College Cork':208, 'University of Limerick':288,
        'Dublin City University':302, 'Technological University Dublin':401}
df = pd.DataFrame(list(irish_unis_dict.items()),columns = ['uni_name','global_score_rank'])
print(df.head())

geometry = {'Trinity College Dublin':(53.3438, -6.2546), 'Royal College of Surgeons in Ireland (RCSI)':(53.3390, -6.2620),'University College Dublin':(53.3067, -6.2210),
             'National University of Ireland, Galway':(53.2792, -9.0617), 'Maynooth University':(53.3845, -6.6011), 'University College Cork':(51.8935, -8.4921), 'University of Limerick':(52.673479,-8.564095),
        'Dublin City University':(53.3861, -6.2564), 'Technological University Dublin':(53.3515, -6.2693)}
df2 = pd.DataFrame(list(geometry.items()),columns = ['uni_name','geometry'])
print(df2.head())

new_df = pd.merge(df, df2, on='uni_name')
print(new_df.head())

#visualising geographical data

import pandas as pd

irl_uni_only_data = pd.DataFrame({'irish_uni_name' :['Trinity College Dublin', 'Royal College of Surgeons in Ireland (RCSI)','University College Dublin',
             'National University of Ireland, Galway', 'Maynooth University', 'University College Cork', 'University of Limerick',
        'Dublin City University', 'Technological University Dublin'],
                                  'score_rank' :[114, 142,158,188,199,208,288,302,401],
                                  'lat':[53.3438, 53.3390,53.3067,53.2792, 53.3845, 51.8935, 52.673479,53.3861, 53.3515] ,
                                  'long':[-6.2546, -6.2620,-6.2210, -9.0617,-6.6011, -8.4921, -8.564095, -6.2564,-6.2693]})

import folium
trinity_col = folium.Map(location=[-6.2546,53.3438], zoom_start = 21)
trinity_col.save("mymap.html")

for i in range(0,len(irl_uni_only_data)):
   folium.Marker(
       location=[irl_uni_only_data.iloc[i]['lat'], irl_uni_only_data.iloc[i]['long']],
      popup=irl_uni_only_data.iloc[i]['irish_uni_name'],
   ).add_to(trinity_col)
trinity_col.save("mymap2.html")
#please note if starting location is incorrect the markers are correctly placed across Ireland

#additional code looking at geojson file on counties in Ireland
irl_map = gpd.read_file('counties.geojson')
irl_map.plot()

irl_map.crs

irl_map.geometry = irl_map.geometry.to_crs(epsg=3857)