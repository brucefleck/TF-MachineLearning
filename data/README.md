## Contenido de esta carpeta

Esta carpeta `data` contiene el dataset original obtenido de Kaggle

Cryptocurrency Historical Prices por SRK

https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory?resource=download 

El dataset consiste de 23 archivos con los datos historicos de 23 criptomonedas. Cada uno tiene el nombramiento de `coin_<Nombre>.csv`

---

El archivo `cryptos_unificado.csv` contiene la data de todos los archivos antieriores unificados en uno solo. Este despues se usara para calcular las variables extras.

El archivo `cryptos_enriquecido.csv` contiene el dataset unificados con variables extras calculados a partir de los datos originales como forma de enriquecer los datos y tener mas variables para poder entrenar nuestros modelos.

Finalmente, el archivo `data_final.csv` contiene el dataset enriquecido mas el calculo del *variable objetivo*. Con este dataset se realizara el analisis EDA y posterior modelado.