# Fake News Classification

En el presente proyecto exploramos e implementamos un clasificador para [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset), el cual consta de una coleccion relativamente grande de noticias falsas y reales en el idioma inglés. 

## Introducción

Las [fake news](https://en.wikipedia.org/wiki/Fake_news) son noticias con información falsa que, con frecuencia, se usa para dañar la reputación de una persona o entidad, así como para hacer dinero a través de la publicidad. Este tipo de artículos ha venido incrementándose gracias a las redes sociales, dado que mediante artículos tendenciosos y sesgos de confirmación se logran viralizar.

La problemática que generan las fake news ha venido tomando relevancia en los últimos años pues reducen el impacto de las verdaderas noticias y gracias a los algoritmos de feed en las redes sociales, se dificulta el acceso orgánico a las noticias reales.

En la actualidad se teme que los algoritmos de inteligencia artificial sean capaces de crear fake news y más aún, en manos equivocadas sean creadas y esparcidas [1]. En este sentido, la necesidad de usar herramientas igualmente poderosas en contra de las fake news se hace cada vez más significativa.

## Estructura de archivos

El desarrollo de este trabajo se ha realizado utilizando jupyter notebooks con el lenguaje Python, cada una de las notebooks se encuentra numerada y nombrada descriptivamente con su contenido:

* [0_Exploracion_inicial](0_Exploracion_inicial.ipynb)
* [1_Exploracion_con_Limpieza](1_Exploracion_con_Limpieza.ipynb)
* [2_Modelos_Clasicos](2_Modelos_Clasicos.ipynb)
* [3_Analisis_por_palabras](3_Analisis_por_palabras.ipynb)
* [4_Red_Neuronal_Fully_Connected](4_Red_Neuronal_Fully_Connected.ipynb)
* [5_Red_Neuronal_Fully_Connected_con_Limpieza](5_Red_Neuronal_Fully_Connected_con_Limpieza.ipynb)
* [6_Redes_recurrentes](6_Redes_recurrentes.ipynb)
* [07_Red_Base_Subject](6_Redes_recurrentes)
* [08_Red_Clean_Subject](08_Red_Clean_Subject.ipynb)
* [09_Red_Clean_Subject_GridSearch](09_Red_Clean_Subject_GridSearch.ipynb)

Por otro lado, se los archivos referentes a la exposición realizada están dispuestos como sigue:

* [Clasificación de Fake News](Clasificación de Fake News.pdf): contiene los slides
* [Graficos](./Graficos): contiene los gráficos obtenidos de las notebooks, incluyendo los que están en los slides

## Conclusiones

Hemos entrenado una red neuronal fully conected y comparado su performance con algoritmos varios de machine learning mas tradicionales como la regresión logística, los arboles de decisión y ensambles de arboles. En este proceso hemos obtenido una exactitud de ```.995``` eliminando palabras tendenciosas de las cuales nos percatamos explorando y analizando los datos. En este sentido, el machine learning es una herramienta factible para ayudar a combatir el esparcimiento de fake news. Sin embargo, creemos necesaria una recolección de datos mas transparente para evitar sesgos en los algoritmos que resulten en el entrenamiento con los mismos.

## Referencias

[[1] OpenAI has published the text-generating AI it said was too dangerous to share](https://www.theverge.com/2019/11/7/20953040/openai-text-generation-ai-gpt-2-full-model-release-1-5b-parameters)