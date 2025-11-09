# Saliency Maps

## Visão Geral

Em redes profundas (como CNNs grandes, ResNets etc.), nem todas as amostras precisam passar por todas as camadas.
- As amostras “fáceis” (ex: imagem muito nítida de um gato) podem ser classificadas com segurança em camadas intermediárias.
- As “difíceis” (ex: gato deitado no escuro) continuam até o final.

Assim, a rede aprende a interromper o processamento cedo se a confiança for suficiente, economizando tempo e energia.
A partir daí temos um problema entre confiança ser diferente de certeza. A saída “softmax” nem sempre é um bom indicador de confiança, pois os modelos podem estar "overconfident".

Aí que entrariam os mapas de saliência, podemos usar eles como métrica de “atenção confiável”

## O que são Mapas de Saliência?

Um **mapa de saliência** é uma representação visual que destaca as regiões mais importantes ou relevantes de uma imagem para um modelo de aprendizado de máquina. Em essência, o objetivo de um mapa de saliência é refletir o grau de importância de um pixel para o modelo.
São ferramentas proeminentes em XAI (Inteligência Artificial Explicável), fornecendo explicações visuais do processo de tomada de decisão de modelos de aprendizado de máquina, especialmente redes neurais profundas. Eles destacam as regiões na entrada (imagens, texto, etc.) que são mais influentes na saída do modelo, indicando onde o modelo está "olhando" ao fazer uma previsão. Se a rede realmente “olha” para a região correta da imagem (segundo o mapa de saliência), então ela está pronta para sair cedo (early exit), ou seja, podemos definir um critério de ativação antecipada baseado em coerência espacial.

Pode fazer isso de algumas maneiras:

1. Medida de concentração espacial
- Você pode medir quão focado é o mapa de saliência:
- Calcule a entropia espacial (quanto mais difuso, menos confiança).
- Calcule o momento de inércia do mapa (se há uma única região dominante ou várias).

2. Medida de coerência intercamadas
- Gere o mapa de saliência em camadas intermediárias.
- Compare-o com o mapa final (usando cosine similarity ou SSIM).
- Se o mapa intermediário já está “olhando para o mesmo lugar” que o final, isso indica que a decisão já amadureceu — pode sair.

## Algoritmos

Existem diversas abordagens para a criação de mapas de saliência.

*   **Saliência Estática**: Baseia-se em características e estatísticas da imagem para localizar regiões de interesse.
*   **Saliência de Movimento**: Utiliza o movimento em um vídeo, detectado por fluxo óptico, onde objetos em movimento são considerados salientes.
*   **Objectness**: Reflete a probabilidade de uma janela de imagem cobrir um objeto, gerando caixas delimitadoras onde um objeto pode estar.
*   **TASED-Net**: Uma rede de codificador-decodificador que extrai características espaço-temporais de baixa resolução e as decodifica para gerar o mapa de saliência.
*   **STRA-Net**: Integra características espaço-temporais via acoplamento de aparência e fluxo óptico, aprendendo saliência multi-escala através de mecanismos de atenção.
*   **STAViS**: Combina informações visuais e auditivas espaço-temporais para localizar fontes sonoras e fundir as saliências.

As Redes Neurais Profundas (DNNs) convencionais geralmente possuem um único ponto de saída, realizando previsões após processar todas as camadas. No entanto, nem todas as entradas exigem a mesma quantidade de computação para alcançar uma previsão confiante. **Early Exit** (ou saída antecipada) é uma técnica que incorpora múltiplos "pontos de saída" em uma arquitetura de DNN, permitindo que a inferência seja interrompida precocemente em pontos intermediários.

Em uma DNN com Early Exit, ramificações laterais são adicionadas em diferentes profundidades da rede principal. Cada ramificação lateral possui um classificador que pode fazer uma previsão. Durante a inferência, a rede avalia a confiança da previsão em cada ponto de saída. Se a confiança atingir um determinado limiar, a inferência é interrompida e a previsão é aceita, sem a necessidade de processar as camadas subsequentes. Para entradas mais complexas, a inferência continua através de mais camadas da rede principal até que um ponto de saída atinja o limiar de confiança ou até que a saída final da rede seja alcançada.

### Extra

Isso era mentira em 1989 e continua sendo hoje em 2025.Os modelos que temos hoje como resnets, llms etc. Contém centenas de camadas, mas o mesmo tempo não temos nenhuma visão prática sobre o que realmente está sendo feito dentro dessas camadas de uma maneira semanticamente útil.

Já conhecemos o tipo de computação que acontece nessas camadas, mas não sabemos quais delas são necessárias e o que elas realmente estão fazendo com o problema.

Para tomadas de decisão, muitas pessoas gostariam de possuir algum grau de interpretabilidade, que podemos simplificar em “olhar por dentro da caixa-preta” do modelo para entender por que ele faz o que faz. Instrospeção dos modelos.

Isso faz com que dentro de processos de tomada de decisão ainda precisem de um capital humano (discutível).

A pergunta é: Esse tipo de instrospeção é possível em Redes neurais profundas?

Se é possível, quais propriedades elas precisam satisfazer
para que sejam interpretáveis?

É claro que se o modelo for arbitratiamente complicado, não há como o ser humano interpretá-lo, então somente se tivermos certas suposições estruturais sobre o modelo podemos falar sobre interpretá-lo.

Um modelo pode ser matematicamente preciso e estatisticamente ótimo, mas tão complexo e não linear que não conseguimos compreender o raciocínio interno dele de forma intuitiva.

Uma rede neural profunda com bilhões de parâmetros (como GPT-5) não é algo que um ser humano possa entender diretamente olhando para pesos e equações. Mesmo que tivéssemos todos os parâmetros do modelo, a estrutura é tão intrincada que é impossível traduzir “por que” ele tomou uma decisão específica em linguagem humana.

Isso significa que, para entender um modelo, precisamos impor restrições ou estruturas conhecidas que tornem sua lógica mais transparente. Essas suposições estruturais podem ser de vários tipos:

Tipo de Suposição
- Linearidade	-> Assume que a relação entre variáveis é linear
- Sparsidade (esparsidade) -> Assume que poucas variáveis são relevantes
- Hierarquia de decisão -> Assume decisões em etapas (if/else)
- Modularidade -> Divide o modelo em blocos compreensíveis	Redes neurais explicáveis por camadas/funções
- Atenção interpretável	-> Impõe que o modelo “explique” o que prioriza

A maneira mais comum de visualizar saliências para uma rede neural, em especial para CNNs, é usar a saliência de gradiente de entrada, onde a importância é essencialmente codificada pela sensibilidade dos pixels.

Se f é a rede neural de valor escalar, ela mapeia a entrada vetorial x para um valor escalar y. Então o gradiente da entrada em relação a saída é o que iremos definir como mapa de saliência.

Podemos visualizar a magnitude dos gradientes que destaca as partes importantes da imagem

Um dos principais problemas de utilizar gradientes para gerar mapas de saliência é que eles dependem das variações locais da função de saída em relação à entrada. Isso significa que regiões planas ou uniformes da imagem, onde a função do modelo muda pouco ou nada , recebem pouca ou nenhuma atribuição, mesmo que sejam semanticamente importantes para a decisão do modelo. Em outras palavras, os gradientes podem ignorar características relevantes simplesmente porque a função é localmente constante nessa região.

Atribuição de importância em métodos de gradiente mede quanto a saída do modelo muda se você alterar um pixel.
Se a saída muda muito ao mexer em um pixel, o modelo considera esse pixel importante.
Se a saída muda pouco ou nada, o pixel é considerado irrelevante, mesmo que faça parte de uma característica visual significativa.
Em outras palavras, a importância não é dada pelo conteúdo semântico da imagem, mas pelo grau de sensibilidade da função do modelo à entrada.

Isso é o que gradientes capturam

Recomendação futura: Fazer um modelo com várias saídas e ver se o mapa de saliência difere

## Resultado do primeiro experimento (CNN_binary.ipynb)

O modelo está aprendendo efetivamente, com redução consistente da perda e aumento da acurácia, finalizando com ~85-90%, o que indica um desempenho razoável. Não há sinais de "underfitting", pois o modelo consegue aprender os padrões dos dados.
O gap crescente entre as perdas de treinamento e validação (especialmente visível após a época 60) sugere que o modelo pode estar começando a memorizar os dados de treinamento em vez de generalizar. As oscilações na curva vermelha de acurácia sugerem que o conjunto de validação pode ser pequeno ou que o modelo é sensível a variações nos dados.

Nesse contexto, foi possível avaliar os diferentes algoritmos baseados em gradiente e esboçar o que de fato vem a ser um mapa de saliência.

## Resultado do segundo experimento (CNN_early_exit.ipynb)

Aqui o modelo performou de maneira mais limitada, mas no final, nosso objetivo era ver como o mapa de saliência mudava conforme avançava nas camadas e isso foi um sucesso.

## Resultado do terceiro experimento (CNN_early_exit.ipynb)

### U-Net

Enquanto CNNs são tipicamente usadas para tarefas de classificação (onde a saída é um único rótulo de classe para uma imagem inteira), em problemas de segmentação biomédica o objetivo é o que chamam de **localização**, que consiste em atribuir um rótulo de classe a cada pixel.

Antes do U-Net, uma abordagem comum era usar uma rede de janela deslizante (sliding-window setup) para prever o rótulo de cada pixel fornecendo uma região local (patch) ao redor dele. Embora eficaz, essa estratégia tinha duas desvantagens principais:
- Lentidão: A rede precisava ser executada separadamente para cada patch, com muita redundância.
- Trade-off: Havia um compromisso entre a precisão da localização e o uso do contexto. Patches maiores exigiam mais camadas de *max-pooling*, reduzindo a precisão da localização.

O U-Net se baseia na arquitetura "fully convolutional network", modificada e estendida para funcionar com muito poucas imagens de treinamento e produzir segmentações mais precisas. (Talvez sirva para nosso artigo?)

### Arquitetura da Rede

O nome "U-Net" deriva de sua forma em U, simétrica. A arquitetura, ilustrada na Figura 1 do artigo, é composta por dois caminhos principais:

1.  **Caminho Contrativo (Contracting Path):** Localizado no lado esquerdo da arquitetura, este caminho tem como objetivo **capturar o contexto**.
    *   Segue a arquitetura típica de uma CNN: aplicação repetida de duas convoluções $3\times3$ (não acolchoadas, ou *unpadded*), cada uma seguida por uma unidade linear retificada (**ReLU**), e uma operação de *max pooling* $2\times2$ (com passo 2) para subamostragem (*downsampling*).
    *   A cada etapa de subamostragem, o número de canais de *feature maps* é dobrado.

2.  **Caminho Expansivo (Expansive Path):** Localizado no lado direito, este caminho é simétrico ao contratante e permite a **localização precisa**.
    *   Cada etapa consiste em uma superamostragem (*upsampling*) do *feature map*, seguida por uma convolução $2\times2$ ("up-convolution") que reduz pela metade o número de canais de *features*.
    *   Em seguida, ocorre uma **concatenação** com o *feature map* correspondente (recortado) do caminho contrativo. Essa etapa combina as informações de alta resolução do caminho contrativo com as informações de contexto propagadas (já que o caminho expansivo também possui um grande número de canais de *features*).
    *   Finaliza com duas convoluções $3\times3$, cada uma seguida por uma ReLU.
    *   O recorte (*cropping*) é necessário devido à perda de pixels de borda em cada convolução.

A rede U-Net não possui camadas totalmente conectadas (*fully connected layers*). Na camada final, uma convolução $1\times1$ é usada para mapear cada vetor de *features* de 64 componentes para o número desejado de classes. A rede possui um total de **23 camadas convolucionais**.

### Estratégia de Treinamento

A rede foi treinada usando a implementação de gradiente descendente estocástico do Caffe.

#### 1. Aumento de Dados (Data Augmentation)

O **aumento excessivo de dados** é crucial para o sucesso da U-Net, pois permite que a rede use as poucas amostras anotadas de forma mais eficiente.
*   É essencial para ensinar a rede a **invariância à rotação e ao deslocamento** e robustez a variações de valor de cinza e, principalmente, a **deformações elásticas**.
*   As deformações elásticas aleatórias dos dados de treinamento são o conceito-chave para treinar a rede com pouquíssimas imagens anotadas, pois simulam as variações mais comuns em tecidos biológicos.

#### 2. Função de Perda Ponderada (Weighted Loss)

Um desafio em muitas tarefas de segmentação celular é a **separação de objetos tocantes** da mesma classe. Para resolver isso, os autores propuseram o uso de uma função de perda ponderada:
*   A função de energia (loss function) é calculada usando um *soft-max* pixel a pixel sobre o *feature map* final combinado com a função de perda de entropia cruzada.
*   É introduzido um **mapa de peso** $w(x)$ para dar mais importância a certos pixels durante o treinamento.
*   O mapa de peso é pré-calculado para:
    *   Compensar a diferente frequência de pixels de uma determinada classe.
    *   Forçar a rede a aprender as pequenas **bordas de separação** introduzidas entre as células tocantes. As labels de fundo (background) que separam células tocantes recebem um grande peso na função de perda.

#### 3. Estratégia de Sobreposição de Tiles (Overlap-tile strategy)

O U-Net usa apenas a parte válida de cada convolução, de modo que o mapa de segmentação de saída contém apenas pixels para os quais o contexto completo estava disponível na imagem de entrada. Essa estratégia permite a **segmentação contínua de imagens arbitrariamente grandes**.
*   Para prever os pixels na região da borda da imagem, o contexto ausente é extrapolado pelo **espelhamento da imagem de entrada**.
*   Esta estratégia de ladrilhamento (*tiling*) é importante para aplicar a rede a imagens grandes, pois o contrário limitaria a resolução pela memória da GPU.

### Bases de Dados (Datasets) e Resultados

Os experimentos demonstraram a aplicação do U-Net em três tarefas distintas de segmentação biomédica.

#### 1. Segmentação de Estruturas Neurais (EM Stacks)

*   **Desafio:** O Desafio de Segmentação EM (EM segmentation challenge), iniciado no ISBI 2012.
*   **Base de Dados:** Consiste em 30 imagens ($512\times512$ pixels) de microscopia eletrônica de transmissão por seção serial (EM) do cordão nervoso ventral (VNC) da larva de primeiro instar de *Drosophila*.
*   **Anotações:** Cada imagem possui um mapa de segmentação de verdade fundamental (*ground truth*) totalmente anotado para **células (branco) e membranas (preto)**.
*   **Avaliação:** Usa o "erro de *warping*", o "erro Rand" e o "erro de pixel".
*   **Resultado do U-Net:** O U-Net superou o método anterior (rede de janela deslizante de Ciresan et al.). Alcançou um erro de *warping* de 0.0003529 (o novo melhor resultado) e um erro Rand de 0.0382.

#### 2. Segmentação Celular em Imagens de Microscopia de Luz (ISBI Cell Tracking Challenge 2015)

O U-Net também foi aplicado a esta tarefa de segmentação celular, que faz parte do desafio ISBI Cell Tracking Challenge.

**Dataset 1: "PhC-U373"**
*   **Descrição:** Células de Glioblastoma-astrocitoma U373 registradas por microscopia de contraste de fase (Phase Contrast Microscopy).
*   **Anotações:** 35 imagens de treinamento parcialmente anotadas.
*   **Resultado do U-Net:** Atingiu uma IOU (*Intersection over Union*) média de **92%**, sendo significativamente melhor do que o segundo melhor algoritmo (83%).

**Dataset 2: "DIC-HeLa"**
*   **Descrição:** Células HeLa em vidro plano registradas por microscopia de contraste de interferência diferencial (DIC - Differential Interference Contrast).
*   **Anotações:** 20 imagens de treinamento parcialmente anotadas.
*   **Resultado do U-Net:** Atingiu uma IOU média de **77.5%**, sendo significativamente melhor do que o segundo melhor algoritmo (46%).

Em essência, a U-Net funciona como um filtro complexo e inteligente que aprende a "desenhar" o contorno das estruturas biológicas. Se o treinamento padrão é como aprender a reconhecer um objeto em uma foto (classificação), a U-Net é como aprender a traçar o contorno exato de cada objeto nessa foto (segmentação), mesmo quando os objetos estão encostados, usando a estratégia de aumento de dados como um "treinamento de elasticidade" para garantir que a rede reconheça as células, independentemente de estarem espremidas ou deformadas.



### Dataset Oxford-IIIT Pet

The dataset contains photographs of cats and dogs across 37 categories, with approximately 200 images per class. The images exhibit significant variation in scale, pose, and lighting conditions. Each image is accompanied by ground-truth annotations, including the breed, head region of interest (ROI), and pixel-level trimap segmentation. This dataset is widely used for <b>image classification</b>, as well as for <b>segmentation</b> and <b>object detection</b> tasks.

| **Task**                 | **Question it answers**                       | **Output**              | **Example use case**                    |
| ------------------------ | --------------------------------------------- | ----------------------- | --------------------------------------- |
| **Image Classification** | What is in the image?                         | One or more labels      | “It’s a Siamese cat.”                   |
| **Object Detection**     | What is in the image and where is it located? | Bounding boxes + labels | Detecting multiple animals in one photo |
| **Image Segmentation**   | What is in each pixel?                        | Pixel-wise mask         | Separating the cat from the background  |

Link: https://www.robots.ox.ac.uk/~vgg/data/pets/

![image.png](attachment:image.png)

##### Pixel-wise mask

A pixel mask is an auxiliary image used to represent which parts of the original image belong to a specific object or class. It is fundamental in image segmentation tasks.
A pixel mask has the same size as the original image (same width and height), but instead of containing real colors (RGB), each pixel stores a numerical value indicating the class that pixel belongs to.

| Pixel | Mask Value       | Meaning                                      |
| ----- | ---------------- | -------------------------------------------- |
| 0     | Background       | Irrelevant area                              |
| 1     | Cat              | Part of the animal                           |
| 2     | Uncertain border | Transition region between background and cat |

Practical Example -> Imagine the following image:
- Original image: A white cat sitting on a sofa.
- Pixel mask: Pixels forming the cat → value 1

Pixels from the background (sofa, wall) → value 0
When visualized, this mask appears as a grayscale or artificially colored image, highlighting only the shape of the object.

Masks allow the model to learn the exact shape and precise boundaries of an object.
Unlike detection (which uses bounding boxes), segmentation shows which pixels belong to the object.

| Type           | Description                                         | Example of use                         |
| -------------- | --------------------------------------------------- | -------------------------------------- |
| **Binary**     | Contains only 0 (background) and 1 (object)         | Segmenting the cat from the background |
| **Multiclass** | Each number represents a different class            | Separating cat, dog, and background    |
| **Trimap**     | Three levels (background, object, uncertain border) | Used in the *Oxford-IIIT Pet* dataset  |
