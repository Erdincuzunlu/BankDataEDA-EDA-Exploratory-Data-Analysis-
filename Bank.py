import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, cross_validate


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1000)

df = pd.read_csv("/Users/erdinc/PycharmProjects/pythonProject3/Geliştirdiklerim/bank.csv")
df.head()
df.shape
df.nunique()

# Age  = yaş
# Job = Meslek
# Marital = Medeni Durum
# Education = Eğitim Durumu
# Default = Kredi kartı ödeme gecikmesi durumu
# Balance Bakiye
# Housing = Konut kredisi
# Loan = Kişisel Kredi
# Contact = İletişim Türü
# Day = Gün
# Month = Ay
# Duration = Son görüşme süresi
# Campaing = İletişim sayısı ( Verilen kampanya sonrası müşteri ile kaç kez görüşüldü )
# pdays = Müşteriyle önceki kampanya gününden sonra kaç gün geçtiğini gösterir / Eğer müşteri ile daha önce iletişim kurulmadıysa bu değer -1 olur
# previous = (Önceki İletişim Sayısı) Müşteriyle önceki kampanyada kaç kez iletişim kurulduğunu gösterir (numerik değer).
# poutcome = (Önceki Kampanyanın Sonucu) Müşterinin daha önce katıldığı kampanyanın sonucunu belirtir.  / success” (başarılı), “failure” (başarısız), “other” (diğer), “unknown” (bilinmiyor)
# deposit =  (Mevduat)  Hedef değişken (bağımlı değişken). Müşterinin vadeli mevduat yatırıp yatırmadığını belirtir. İki kategoriye sahiptir / Yes ( mevduat yaptı ) No ( mevduat yapmadı )



# İlk olarak veri seti üzerinde
# genel bir veri keşif (exploratory data analysis - EDA) yapacağız.
# Bu, veriyi daha iyi anlamamıza ve modelleme öncesinde hangi veriyi
# temizleme ve ön işleme adımlarını yapmamız gerektiğini görmemize yardımcı olur.

def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Aykırı değerler için alt ve üst sınırları hesaplar.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Aykırı değerlerin sınırlarının hesaplanacağı veri seti.
    col_name: str
        Aykırı değerleri kontrol edilecek değişkenin ismi.
    q1: float, optional
        Birinci çeyreklik değeri, varsayılan 0.05.
    q3: float, optional
        Üçüncü çeyreklik değeri, varsayılan 0.95.

    Returns
    -------
    low_limit: float
        Aykırı değer alt sınırı.
    up_limit: float
        Aykırı değer üst sınırı.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

# Check Outlier
def check_outlier(dataframe, col_name):
    """
    Bir değişkende aykırı değer olup olmadığını kontrol eder.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Veri seti.
    col_name: str
        Aykırı değerleri kontrol edilecek değişkenin ismi.

    Returns
    -------
    bool
        Eğer aykırı değer varsa True, yoksa False döner.
    """
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    return not outliers.empty

# Replace With Threshold
def replace_with_threshold(dataframe, variable):
    """
    Aykırı değerlere sahip gözlemleri alt ve üst limitlerle sınırlandırır.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Veri seti.
    variable: str
        Aykırı değerleri düzeltilecek değişkenin ismi.
    """
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Grab Column Names
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik ama kardinal değişkenlerin isimlerini döner.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Değişkenlerin alınacağı veri seti.
    cat_th: int, optional
        Numerik görünümlü kategorik değişkenler için sınıf eşik değeri. Varsayılan 10.
    car_th: int, optional
        Kardinal kategorik değişkenler için sınıf eşik değeri. Varsayılan 20.

    Returns
    -------
    cat_cols: list
        Kategorik değişken isim listesi.
    num_cols: list
        Numerik değişken isim listesi.
    cat_but_car: list
        Kardinal kategorik değişken isim listesi.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# Cat Summary
def cat_summary(dataframe, col_name, plot=False):
    """
    Kategorik değişkenlerin frekans ve oranlarını gösterir, isteğe bağlı olarak grafiğini çizer.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Veri seti.
    col_name: str
        Özet istatistiklerinin çıkarılacağı kategorik değişken ismi.
    plot: bool, optional
        Grafik çizilip çizilmeyeceğini belirler. Varsayılan False.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# Num Summary
def num_summary(dataframe, col_name, plot=False):
    """
    Numerik değişkenlerin temel istatistiksel özetini ve isteğe bağlı olarak grafiğini gösterir.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Veri seti.
    col_name: str
        Özet istatistiklerinin çıkarılacağı numerik değişken ismi.
    plot: bool, optional
        Grafik çizilip çizilmeyeceğini belirler. Varsayılan False.
    """
    quantiles = [0.25, 0.5, 0.75, 0.99]
    print(dataframe[col_name].describe(quantiles).T)
    print("##########################################")
    if plot:
        dataframe[col_name].hist(bins=20)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show()

# Target Summary with Numerical
def num_summary(dataframe, col_name, plot=False):
    """
    Numerik değişkenlerin temel istatistiksel özetini ve isteğe bağlı olarak grafiğini gösterir.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Veri seti.
    col_name: str
        Özet istatistiklerinin çıkarılacağı numerik değişken ismi.
    plot: bool, optional
        Grafik çizilip çizilmeyeceğini belirler. Varsayılan False.
    """
    quantiles = [0.25, 0.5, 0.75, 0.99]
    print(dataframe[col_name].describe(quantiles).T)
    print("##########################################")
    if plot:
        dataframe[col_name].hist(bins=20)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show()

#Target Summary with Categorical
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Hedef değişken ile kategorik değişken arasındaki ilişkiyi özetler.

    Parameters
    ----------
    dataframe: pandas DataFrame
        Veri seti.
    target: str
        Hedef değişken.
    categorical_col: str
        İncelenecek kategorik değişkenin ismi.
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}))
    print("##########################################")
# Bu değişkenler üzerinde veri analizi, ön işleme ve modelleme işlemleri gerçekleştirebiliriz.
# Özellikle “deposit” değişkeni bağımlı değişken olup,
# müşteri vadeli mevduat yapıp yapmadığını tahmin etmeye yönelik sınıflandırma problemleri için kullanılır.


# Adım 1. Veri Setini Anlama ve Genel Bakış
df.head()
df.info()
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   age        11162 non-null  int64
#  1   job        11162 non-null  object
#  2   marital    11162 non-null  object
#  3   education  11162 non-null  object
#  4   default    11162 non-null  object
#  5   balance    11162 non-null  int64
#  6   housing    11162 non-null  object
#  7   loan       11162 non-null  object
#  8   contact    11162 non-null  object
#  9   day        11162 non-null  int64
#  10  month      11162 non-null  object
#  11  duration   11162 non-null  int64
#  12  campaign   11162 non-null  int64
#  13  pdays      11162 non-null  int64
#  14  previous   11162 non-null  int64
#  15  poutcome   11162 non-null  object
#  16  deposit    11162 non-null  object
# dtypes: int64(7), object(10) Verimizde 7 adet int gözlem 10 adet object gözlem bulunmaktadır.


# Adım 1. 2  : Hedef değişkenin (deposit) dağılımını inceleyelim

df["deposit"].value_counts()

# no     5873
# yes    5289

df["deposit"].value_counts(normalize=True) # oransal dağılımına bakalım
# no    0.526
# yes   0.474
df["deposit"].nunique()
# hedef değişkenin sınıf dağılımını inceledik. Bu dağılımdan görebildiğimiz kadarıyla,
# “deposit” değişkeni iki sınıftan oluşuyor ve sınıflar birbirine oldukça yakın oranlarda dağıtılmış (%52 “no”, %47 “yes”).

# Adım 2 Veri Keşfi
# 2. 1 Kategorik değişkenleri yakalamak
# Bunun için daha önceden hazırladığım fonksiyonları çağırmak istiyorum

# Kategorik ve sayısal değişkenleri belirle
cat_cols, num_cols, cat_but_car = grab_col_names(df)

print("Kategorik Değişkenler:")
print(cat_cols)

print("Sayısal Değişkenler:")
print(num_cols)

cat_summary(df, "job", plot=True)

# Yüksek Oranlı Gruplar: Yönetim, mavi yaka ve teknik meslek grupları, bankanın en büyük müşteri tabanını oluşturuyor.
# Bu durum, bankanın yüksek gelir grubuna daha iyi hizmet verebileceği anlamına geliyor.

# Düşük Oranlı Gruplar: Öğrenci, işsiz ve ev hizmetçileri gibi gruplar,
# bankanın hedef kitlesinin daha düşük gelir düzeyine sahip olduğunu gösteriyor.
# Bankanın, bu gruplara özel ürünler geliştirmesi faydalı olabilir.
# Hedef Kitle Stratejisi: Banka, yönetim ve mavi yaka çalışanları gibi yüksek oranlı gruplara yönelik kampanyalar düzenleyebilir.
# Ayrıca, daha düşük oranlı gruplara yönelik özel teklifler sunarak müşteri tabanını genişletebilir.

cat_summary(df, "marital", plot=True)

# Evli bireyler en yüksek orana sahip, yani bu veri setindeki bireylerin çoğunluğu evli.
# Bu, muhtemelen bankanın sunduğu ürün veya hizmetlerin, evli bireyler arasında daha fazla talep gördüğünü gösterebilir.

cat_summary(df, "education", plot=True)
# education
# secondary       5476 49.059
# tertiary        3689 33.050
# primary         1500 13.438
# unknown          497  4.453
# Eğitim durumu ile finansal durum, gelir düzeyi ve kredi alma potansiyeli arasında genellikle güçlü bir ilişki vardır.

cat_summary(df, "default", plot=True)
# no         10994 98.495
# yes          168  1.505
# Banka müşterilerinin büyük çoğunluğunun ödeme geçmişi iyi olduğundan,
# bu müşteri kitlesi banka açısından düşük riskli kabul edilebilir.
# Biraz daha derinlemesine analiz etmek istersek

df.groupby("default")["balance"].mean()
# no    1552.841
# yes    -61.804

sns.barplot(x="default", y="balance", data=df)
plt.show()

# Eğer daha analitik bir bakış açısıyla yaklaşmak istiyorsak, “default” yapan ve yapmayan grupların
# bakiyeleri arasında istatistiksel olarak anlamlı bir fark olup olmadığını test edebiliriz.

# default yapanlar ve yapmayanlar

no_default = df[df["default"] == "no"]["balance"]
yes_default = df[df["default"] == "yes"]["balance"]
from scipy.stats import ttest_ind
t_test, p_val = ttest_ind(no_default, yes_default)
print(t_test, p_val)
# h0 : iki grup arasında anlamlı bir fark yoktur
# H1 : iki grup arasında anlamlı bir fark vardır.

# 	p-değeri: 1.15e-10 (yani 0.000000000115, ki bu 0.05’ten çok küçük)
# H0 hipotezi red edilir . p_value < 0.05
# iki grup arasında anlamlı bir fark olduğunu kabul ederiz.

cat_summary(df, "housing", plot=True)
# housing
# no          5881 52.688
# yes         5281 47.312

cat_summary(df, "loan", plot=True)
# loan
# no    9702 86.920
# yes   1460 13.080
# burada şunu düşündüm. Hem ev kredisi olan hem biresysel kredisi olanlar arasında bir ilişki var mı diye bakmak istiyorum
pd.crosstab(df['housing'], df['loan'])

cat_summary(df, "contact", plot=True)
# contact
# cellular      8042 72.048
# unknown       2346 21.018
# telephone      774  6.934
# unknown kategorisinde %21.018’lik bir oran mevcut. Bu, bazı durumlarda iletişim bilgileri eksik veya bilinmiyor olabilir.
# Bu, banka için potansiyel bir sorun olabilir, çünkü iletişim kurulamayan müşteriler kaybedilebilir.

cat_summary(df, "month", plot=True)
# Bankanın bu dönemdeki pazarlama stratejileri, müşteri sayısını artırmış olabilir. ( MAY )

cat_summary(df, "poutcome", plot=True)
#           poutcome  Ratio
# poutcome
# unknown       8326 74.592
# failure       1228 11.002
# success       1071  9.595
# other          537  4.811 Burada bankanın sunduğu kampanyada sadece %9.5 oranla müşterilerine ulaşım sağlamış. Çoğunluk unknown yani
# muhtemelen banka müşterilere ulaşmış olabilir kampanya aracılığı ile fakat yeterli veriyi elinde tutamadığını gösteriyor

cat_summary(df, "deposit", plot=True)
# deposit
# no          5873 52.616
# yes         5289 47.384
# müşterilerin önemli bir kısmının banka mevduat hesabı açtığını gösteriyor.
# Bu, bankanın sunduğu ürün ve hizmetlerin müşteriler arasında kabul gördüğünü ortaya koyuyor.

# Ürün Geliştirme: Mevduat hesabı açmayanlar için cazip koşullar veya alternatif ürünler sunarak
# bu grubun banka ile olan ilişkisini artırmak mümkün olabilir.

df.isnull().sum() # Eksik veri bulunmamaktadır.

# Aykırı değerlerin kontrol edilmesi

for col in num_cols:
    check_outlier(df, col)

for col in num_cols:
    print(f"{col} Değerleri:\n", df[col].describe())

for col in num_cols:
    if check_outlier(df, col):
        print(f"{col} sütununda aykırı değer var.")
    else:
        print(f"{col} sütununda aykırı değer yok.")

# age sütununda aykırı değer yok.
# balance sütununda aykırı değer var.
# day sütununda aykırı değer yok.
# duration sütununda aykırı değer var.
# campaign sütununda aykırı değer var.
# pdays sütununda aykırı değer var.
# previous sütununda aykırı değer var.


plt.figure(figsize=(10, 6))
sns.boxplot(x=df['balance'])
plt.title('Balance Aykırı Değer Analizi')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['duration'])
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['campaign'])
plt.show()

# # Aykırı değerleri düzeltme işlemi
columns_with_outliers = ["campaign", "duration", "balance", "pdays", "previous"]
for col in columns_with_outliers:
    replace_with_threshold(df, col)

# aykırı değerlere uyguladığımız düzeltme işlemi sonra tekrar bakıyoruz aykırı değer var mı yokmu ?

for col in columns_with_outliers:
    has_outliers = check_outlier(df, col)
    print(has_outliers)

# Temizlenmiş verilerin özet istatistikleri
df.describe().T

# Adım 3 .Kategorik Değişkenlerin Dönüştürülmesi (Encoding)

# 3. 1 Label Encoding

binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]

label_encoding= LabelEncoder()
for col in binary_cols:
    df[col] = label_encoding.fit_transform(df[col])

# 3 2. One hot encoding

df = pd.get_dummies(df, drop_first=True) # Kategorik değişkenleri one hot encoding ile dönüştürme

X = df.drop("deposit_yes", axis=1) # Hedef değişkeni çıkarıyoruz
Y = df["deposit_yes"] # hedef değişkeni seçiyoruz


# Şimdi modelleme sürecine geçebiliriz
# 1 veri kümesini eğitim ve veri seti olarak ayırma

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# # Veriyi standartlaştırma

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Lojistik regresyon modelini oluşturma
model = LogisticRegression()
model.fit(x_train, y_train)

# Burada hata alıyorum. ojistik Regresyon modelinin belirli bir iterasyon sayısına ulaşmadan eğitimi tamamlayamadığını gösteriyor.
# Bu durum genellikle modelin daha fazla iterasyona ihtiyaç duyduğunda ortaya çıkar
from sklearn.preprocessing import StandardScaler

# Verileri standartlaştırma
scaler = StandardScaler()

# Eğitim verisini standartlaştır
x_train_scaled = scaler.fit_transform(x_train)

# Test verisini aynı ölçeklendirme ile dönüştür
x_test_scaled = scaler.transform(x_test)

# Lojistik regresyon modelini oluşturma
model = LogisticRegression(max_iter=1000)  # max_iter'i artırdık
model.fit(x_train_scaled, y_train)

# Modeli eğitiğimize göre
# Test setinde tahmin yapma
y_pred = model.predict(x_test_scaled)
y_pred_proba = model.predict_proba(x_test_scaled)

# Modelin performansını değerlendirelim şimdide ....

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error
)

# Test setinde tahmin yapma
y_pred = model.predict(x_test_scaled)
y_pred_proba = model.predict_proba(x_test_scaled)[:, 1]  # Pozitif sınıf için olasılıkları al

# Değerlendirme metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)  # Pozitif sınıf için olasılıkları kullan
mse = mean_squared_error(y_test, y_pred.astype(int))  # y_pred'i int'e dönüştür
rmse = mean_squared_error(y_test, y_pred.astype(int), squared=False)  # RMSE

# Sonuçları yazdırma
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Accuracy: 0.8061 ###  Model, test setindeki örneklerin yaklaşık %80.61’ini doğru bir şekilde sınıflandırabilmiştir.
# Precision: 0.8096 ### Modelin pozitif sınıfa tahmin ettiği değerlerin %80.96’sı gerçekten pozitif. Yani, modelin pozitif tahminleri güvenilir görünüyor
# Recall: 0.7769 ### Model, gerçek pozitiflerin %77.69’unu doğru tahmin edebilmiştir. Bu, modelin pozitif sınıfları bulma konusundaki başarısını gösterir
# F1 Score: 0.7929 ### F1 skoru, precision ve recall değerlerinin harmonik ortalamasıdır. Bu değer, modelin hem doğruluğunu hem de duyarlılığını dengeleyerek iyi bir performans gösterdiğini belirtir
# ROC AUC: 0.9006 ### ROC AUC, modelin pozitif ve negatif sınıfları ayırt etme yeteneğini gösterir.
# MSE: 0.1939 ### MSE, modelin tahminlerinin ne kadar hata payı ile gerçekleştiğini ölçer. Düşük bir MSE değeri, modelin tahminlerinin gerçeğe yakın olduğunu gösterir.
# RMSE: 0.4404 ### RMSE, tahmin hatalarının karekökü alınarak elde edilen bir ölçüdür. Düşük bir RMSE değeri, modelin genel hata oranının düşük olduğunu gösterir