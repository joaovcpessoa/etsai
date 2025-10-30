# Week 5

Em redes profundas (como CNNs grandes, ResNets etc.), nem todas as amostras precisam passar por todas as camadas.
- As amostras “fáceis” (ex: imagem muito nítida de um gato) podem ser classificadas com segurança em camadas intermediárias.
- As “difíceis” (ex: gato deitado no escuro) continuam até o final.

Assim, a rede aprende a interromper o processamento cedo se a confiança for suficiente, economizando tempo e energia.
A partir daí temos um problema entre confiança ser diferente de certeza. A saída “softmax” nem sempre é um bom indicador de confiança, pois os modelos podem estar "overconfident".

Aí que entrariam os mapas de saliência, podemos usar eles como métrica de “atenção confiável”

### O que são Mapas de Saliência?

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

### Algoritmos

Existem diversas abordagens para a criação de mapas de saliência.

*   **Saliência Estática**: Baseia-se em características e estatísticas da imagem para localizar regiões de interesse.
*   **Saliência de Movimento**: Utiliza o movimento em um vídeo, detectado por fluxo óptico, onde objetos em movimento são considerados salientes.
*   **Objectness**: Reflete a probabilidade de uma janela de imagem cobrir um objeto, gerando caixas delimitadoras onde um objeto pode estar.
*   **TASED-Net**: Uma rede de codificador-decodificador que extrai características espaço-temporais de baixa resolução e as decodifica para gerar o mapa de saliência.
*   **STRA-Net**: Integra características espaço-temporais via acoplamento de aparência e fluxo óptico, aprendendo saliência multi-escala através de mecanismos de atenção.
*   **STAViS**: Combina informações visuais e auditivas espaço-temporais para localizar fontes sonoras e fundir as saliências.

As Redes Neurais Profundas (DNNs) convencionais geralmente possuem um único ponto de saída, realizando previsões após processar todas as camadas. No entanto, nem todas as entradas exigem a mesma quantidade de computação para alcançar uma previsão confiante. **Early Exit** (ou saída antecipada) é uma técnica que incorpora múltiplos "pontos de saída" em uma arquitetura de DNN, permitindo que a inferência seja interrompida precocemente em pontos intermediários.

Em uma DNN com Early Exit, ramificações laterais são adicionadas em diferentes profundidades da rede principal. Cada ramificação lateral possui um classificador que pode fazer uma previsão. Durante a inferência, a rede avalia a confiança da previsão em cada ponto de saída. Se a confiança atingir um determinado limiar, a inferência é interrompida e a previsão é aceita, sem a necessidade de processar as camadas subsequentes. Para entradas mais complexas, a inferência continua através de mais camadas da rede principal até que um ponto de saída atinja o limiar de confiança ou até que a saída final da rede seja alcançada.

# Extra

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


## Resultado do primeiro experimento

### Acurácia por Época

- Tendência geral: Ambas as curvas apresentam um crescimento consistente da acurácia, partindo de aproximadamente 0.50-0.52 (50%) e atingindo valores entre 0.85-0.90 (85-90%) ao final do treinamento.
- Comportamento inicial (épocas 0-40): Crescimento rápido e acentuado da acurácia, com a curva de validação (vermelho) apresentando maior volatilidade (oscilações mais pronunciadas) que a de treinamento (azul).
- Comportamento intermediário/final (épocas 40-100): As curvas convergem e se estabilizam em torno de 85-90%, com oscilações moderadas. As duas curvas se entrelaçam frequentemente, sem um gap consistente.
- Volatilidade da validação: A acurácia de validação apresenta picos e quedas mais acentuados ao longo de todo o treinamento, o que é esperado, pois o conjunto de validação é menor e mais sensível a variações.

### Perda por Época

- Tendência geral: Ambas as curvas (treinamento em azul e validação em vermelho) apresentam uma redução consistente da perda ao longo das épocas, partindo de aproximadamente 0.70 e chegando a valores entre 0.25-0.35 ao final.
- Comportamento inicial (épocas 0-40): Há uma queda acentuada e relativamente suave na perda, indicando que o modelo está aprendendo rapidamente os padrões dos dados.
- Comportamento intermediário/final (épocas 40-100): A perda continua diminuindo, mas de forma mais gradual, com maior oscilação na curva de validação.
- Gap entre treinamento e validação: A partir da época 40, nota-se que a perda de validação (vermelho) se mantém consistentemente acima da perda de treinamento (azul), e essa diferença aumenta ligeiramente nas épocas finais. Isso sugere um início de overfitting (sobreajuste), onde o modelo está se ajustando melhor aos dados de treinamento do que aos dados de validação.

### Resumo:
O modelo está aprendendo efetivamente, com redução consistente da perda e aumento da acurácia, finalizando com ~85-90%, o que indica um desempenho razoável. Não há sinais de "underfitting", pois o modelo consegue aprender os padrões dos dados.

O gap crescente entre as perdas de treinamento e validação (especialmente visível após a época 60) sugere que o modelo pode estar começando a memorizar os dados de treinamento em vez de generalizar. As oscilações na curva vermelha de acurácia sugerem que o conjunto de validação pode ser pequeno ou que o modelo é sensível a variações nos dados

### Melhorias:

No geral, o treinamento parece estar progredindo adequadamente, mas há espaço para otimização para melhorar a generalização do modelo.

Melhorias mapeadas:
- Considerar early stopping (parada antecipada) em torno da época 60-70, quando a perda de validação começa a se estabilizar
- Aplicar técnicas de regularização (dropout, weight decay) para reduzir o overfitting
- Avaliar se o conjunto de validação é representativo e suficientemente grande
- Considerar data augmentation para aumentar a diversidade dos dados de treinamento