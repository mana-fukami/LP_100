import re

str1="{{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />"
str2="[[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギ リスの国章]]"
str3="（[[イギリス の国章|国章]]）"
str4="{{lang|fr|[[Dieu et mon droit]]}}<br />（[[ フランス語]]:[[Dieu et mon droit|神と我が権利]]）"
str5=" [[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />''神よ女王を護り賜え''<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}"

new_str=re.sub("'*?","",str2)
new_str=re.sub("\{\{.*?\|.*?\|","",new_str)
new_str=re.sub("\{\{.*?\|","",new_str)
new_str=re.sub("\{\{","",new_str)
new_str=re.sub("\}\}","",new_str)
new_str=re.sub("\[\[ファイル:.*?\|.*?\|","",new_str)
new_str=re.sub("\[\[([^\[])*?\|([^\[])*?\|","",new_str)
new_str=re.sub("\[\[([^\[])*?\|","",new_str)
new_str=re.sub("\[\[","",new_str)
new_str=re.sub("\]\]","",new_str)
new_str=re.sub("\<ref.*?\>.*?</ref>","",new_str)
new_str=re.sub("\<.*?/\>","",new_str)

print(str2)
print(new_str)