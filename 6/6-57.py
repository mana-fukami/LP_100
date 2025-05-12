#無関係の下位問題を抽出する
#短いコードを書く

# k-meansクラスタリング
# 国名に関する単語ベクトルを抽出し、k-meansクラスタリングをクラスタ数k=5として実行せよ。

from sklearn.cluster import KMeans
from WordVector import Word2Vec

w2v=Word2Vec()

file=open("6/country-names.txt","r",encoding="utf-8")
lines=file.readlines()

country=[]
for line in lines:
    country.append(line.replace("\n",""))

country_vec=[]
country_name=[]
vec=None
for name in country:
    vec=w2v.get_vector(name)
    if vec is not None:
        country_vec.append(vec)
        country_name.append(name)

kmeans=KMeans(n_clusters=5,random_state=0)
kmeans.fit(country_vec)

for country,label in zip(country_name,kmeans.labels_):
    print(f"{country}: Cluster {label}")

# 実行結果
"""
Iceland: Cluster 1
Ireland: Cluster 1
Azerbaijan: Cluster 1
Afghanistan: Cluster 1
United Arab Emirates: Cluster 1
Algeria: Cluster 3
Argentina: Cluster 0
Albania: Cluster 1
Armenia: Cluster 1
Angola: Cluster 3
Andorra: Cluster 1
Yemen: Cluster 3
Israel: Cluster 1
Italy: Cluster 1
Iraq: Cluster 1
Iran: Cluster 1
India: Cluster 1
Indonesia: Cluster 4
Uganda: Cluster 3
Ukraine: Cluster 1
Uzbekistan: Cluster 1
Uruguay: Cluster 0
United Kingdom: Cluster 1
Ecuador: Cluster 0
Egypt: Cluster 1
Estonia: Cluster 1
Ethiopia: Cluster 3
Eritrea: Cluster 3
El Salvador: Cluster 0
Australia: Cluster 4
Austria: Cluster 1
Oman: Cluster 1
Netherlands: Cluster 1
Ghana: Cluster 3
Cabo Verde: Cluster 2
Guyana: Cluster 4
Kazakhstan: Cluster 1
Qatar: Cluster 1
Canada: Cluster 1
Gabon: Cluster 2
Cameroon: Cluster 3
Gambia: Cluster 3
Cambodia: Cluster 4
Guinea: Cluster 2
Cyprus: Cluster 1
Cuba: Cluster 0
Greece: Cluster 1
Kiribati: Cluster 4
Kyrgyzstan: Cluster 1
Guatemala: Cluster 0
Kuwait: Cluster 1
Cook Islands: Cluster 4
Grenada: Cluster 4
Croatia: Cluster 1
Kenya: Cluster 3
Costa Rica: Cluster 0
Comoros: Cluster 2
Colombia: Cluster 0
Congo: Cluster 3
Saudi Arabia: Cluster 1
Samoa: Cluster 4
Zambia: Cluster 3
San Marino: Cluster 1
Sierra Leone: Cluster 3
Djibouti: Cluster 2
Jamaica: Cluster 4
Georgia: Cluster 1
Singapore: Cluster 1
Zimbabwe: Cluster 3
Switzerland: Cluster 1
Sweden: Cluster 1
Sudan: Cluster 3
Spain: Cluster 1
Suriname: Cluster 4
Sri Lanka: Cluster 4
Slovakia: Cluster 1
Slovenia: Cluster 1
Seychelles: Cluster 2
Equatorial Guinea: Cluster 2
Senegal: Cluster 3
Saint Lucia: Cluster 4
Somalia: Cluster 3
Solomon Islands: Cluster 4
Thailand: Cluster 1
Tajikistan: Cluster 1
Czechia: Cluster 1
Chad: Cluster 1
China: Cluster 1
Tunisia: Cluster 3
Chile: Cluster 0
Tuvalu: Cluster 4
Denmark: Cluster 1
Germany: Cluster 1
Togo: Cluster 3
Dominican Republic: Cluster 0
Dominica: Cluster 4
Turkmenistan: Cluster 1
Turkey: Cluster 1
Tonga: Cluster 4
Nigeria: Cluster 3
Nauru: Cluster 4
Namibia: Cluster 3
Niue: Cluster 4
Nicaragua: Cluster 0
Niger: Cluster 3
Japan: Cluster 1
New Zealand: Cluster 4
Nepal: Cluster 4
Norway: Cluster 1
Bahrain: Cluster 1
Haiti: Cluster 0
Pakistan: Cluster 1
Holy See: Cluster 1
Panama: Cluster 0
Vanuatu: Cluster 4
Bahamas: Cluster 4
Palau: Cluster 4
Paraguay: Cluster 0
Barbados: Cluster 4
Hungary: Cluster 1
Bangladesh: Cluster 4
Fiji: Cluster 4
Philippines: Cluster 4
Finland: Cluster 1
Bhutan: Cluster 4
Brazil: Cluster 0
France: Cluster 1
Bulgaria: Cluster 1
Burkina Faso: Cluster 3
Brunei Darussalam: Cluster 4
Burundi: Cluster 3
Viet Nam: Cluster 1
Benin: Cluster 3
Venezuela: Cluster 0
Belarus: Cluster 1
Belize: Cluster 4
Peru: Cluster 0
Belgium: Cluster 1
Poland: Cluster 1
Botswana: Cluster 3
Bolivia: Cluster 0
Portugal: Cluster 1
Honduras: Cluster 0
Marshall Islands: Cluster 4
Madagascar: Cluster 2
Malawi: Cluster 3
Mali: Cluster 3
Malta: Cluster 1
Malaysia: Cluster 1
Micronesia: Cluster 4
South Africa: Cluster 3
Myanmar: Cluster 3
Mexico: Cluster 0
Mauritius: Cluster 2
Mauritania: Cluster 2
Mozambique: Cluster 3
Monaco: Cluster 1
Maldives: Cluster 4
Morocco: Cluster 1
Mongolia: Cluster 1
Montenegro: Cluster 1
Jordan: Cluster 1
Latvia: Cluster 1
Lithuania: Cluster 1
Libya: Cluster 3
Liechtenstein: Cluster 1
Liberia: Cluster 3
Romania: Cluster 1
Luxembourg: Cluster 1
Rwanda: Cluster 3
Lesotho: Cluster 3
Lebanon: Cluster 1
"""