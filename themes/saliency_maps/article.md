# NeuLens: Aceleração Dinâmica Espacial de Redes Neurais Convolucionais na Borda

## SUMMARY

Redes neurais convolucionais (CNNs) desempenham um papel importante nos
sistemas de computação móvel e de borda atuais para tarefas baseadas em visão
como classificação e detecção de objetos. No entanto, os métodos de ponta para aceleração de CNNs estão limitados por uma aceleração prática de latência restrita em plataformas de computação geral ou por uma aceleração de latência com perda significativa de precisão. Neste artigo, propomos uma
estrutura de aceleração dinâmica de CNNs baseada em espaço, NeuLens, para
plataformas móveis e de borda. Especificamente, projetamos um novo mecanismo de inferência dinâmica, a super-rede de convolução com reconhecimento de região (ARAC), que remove o máximo possível de operações redundantes dentro dos modelos de CNN com base em redundância espacial e fatiamento de canais.

Na super-rede ARAC, o fluxo de inferência da CNN é dividido em múltiplos microfluxos independentes, e o custo computacional de cada um pode ser ajustado autonomamente com base no conteúdo de entrada em mosaico e nos requisitos da aplicação. Esses microfluxos podem ser carregados em hardware, como GPUs, como modelos individuais. Consequentemente, a redução de sua operação pode ser bem traduzida em aceleração de latência e é compatível com acelerações em nível de hardware. Além disso, a precisão da inferência pode ser bem preservada pela identificação de regiões críticas nas imagens e seu processamento na resolução original com microfluxos grandes.

Com base em nossa avaliação, o NeuLens supera os métodos de referência em até 58% de redução de latência com a mesma precisão e em até 67,9% de melhoria na precisão sob as mesmas restrições de latência/memória.

## INTRO

Tarefas relacionadas à visão computacional geralmente exigem um grande número de recursos computacionais [23]. Muitos estudos se concentram em reduzir o custo computacional da inferência de CNN. Alguns trabalhos propõem arquiteturas de rede leves, como MobileNets [25, 26, 63], CondenseNet [29], ShuffleNets [51, 93] e EfficientNet [71]. Outros estudos comprimem redes existentes por meio de poda [42, 44, 49, 50] ou quantização [32, 33, 57]. Trabalhos recentes propõem várias maneiras que permitem o ajuste dinâmico do custo computacional da inferência de CNN [19,
43] (detalhes na Seção 2.2). Inspirados pela visão humana, onde apenas uma parte limitada da cena visual é processada pelo sistema visual, trabalhos recentes exploram o potencial de redução do custo computacional com base em informações espaciais de entrada, propondo arquiteturas de rede especializadas [83, 88] ou projetando fluxos de computação compatíveis com arquiteturas gerais de CNN [20, 77, 95]. Em streaming de vídeo e
análise, as regiões de interesse (RoIs) são determinadas por
rastreamento entre quadros (Edge-Assisted [47] e Elf [92]) ou por detecção de baixa resolução (DDS [9]). Por meio da codificação baseada em RoI, os tamanhos dos dados de transmissão dos quadros descarregados são significativamente reduzidos [47].

Neste artigo, propomos uma estrutura adaptativa, NeuLens, para aceleração dinâmica da inferência de CNN em dispositivos móveis e de borda.

Primeiramente, projetamos um novo mecanismo dinâmico, a super-rede convolucional com reconhecimento de região (ARAC) (§4), que reduz efetivamente o custo de inferência com pequena perda de precisão. Uma super-rede ARAC é um conjunto de redes com divisão espacial. Ela seleciona adaptativamente sub-redes com tamanhos diferentes para blocos divididos de uma imagem com base em sua relevância para a previsão final. Além disso, projetamos um controlador online leve, DEMUX (§5), que ajusta dinamicamente a seleção de sub-redes por bloco e as configurações da super-rede com base em objetivos de nível de serviço (SLOs) em aplicações reais. Finalmente, avaliamos de forma abrangente a super-rede ARAC em diferentes plataformas móveis/de borda e várias aplicações (§7). Com base em nossa avaliação, a super-rede ARAC alcança uma melhoria de precisão de até 67,9% em relação aos métodos de inferência dinâmica de última geração (SOTA) sob as mesmas restrições de latência/memória (§7.2) e uma precisão até 1,23 vezes maior em relação às técnicas de compressão de modelos SOTA com a mesma latência de inferência (§7.4). Além disso, a aplicação da super-rede ARAC em sistemas de detecção contínua de objetos aumenta o desempenho em até 7,7 vezes em relação às técnicas SOTA [47] (§7.6).

Resumimos as contribuições deste artigo da seguinte forma:

Desenvolvimento de um novo mecanismo de aceleração de CNN para plataformas de computação móvel/de borda (§4). Ao explorar a redundância espacial e de profundidade em imagens e em CNNs, propomos um mecanismo de aceleração, a super-rede ARAC, que reduz efetivamente o consumo de recursos computacionais com uma pequena redução na precisão. Comparada a trabalhos de aceleração existentes, a super-rede ARAC alcança a otimalidade de Pareto na relação entre precisão e latência. Destacamos as seguintes técnicas avançadas na super-rede ARAC:

- Construção de uma super-rede ARAC que se aplica geralmente a arquiteturas de CNN (§4.1). Ao dividir uma imagem de entrada em blocos, a super-rede utiliza sub-redes com diferentes níveis de compressão para analisá-los. As saídas da super-rede são concatenadas e alimentadas nas demais camadas do modelo de CNN para calcular os resultados finais. Essa estrutura permite que a super-rede reduza a redundância espacial e de profundidade na computação sem afetar os esquemas de funcionamento geral das CNNs originais.
- Ajuste de custo computacional por bloco com reconhecimento de conteúdo na super-rede ARAC (§4.4). Um mecanismo de controle de compressão é projetado para analisar efetivamente o conteúdo de cada bloco e atribuir uma sub-rede com o nível de compressão adequado para analisá-los na super-rede. Uma regra de rotulagem é proposta para automatizar a geração do conjunto de treinamento para o mecanismo de controle de compressão.

Conversão eficaz da redução da redundância de operação para a aceleração da latência no dispositivo. Na super-rede ARAC, o fluxo de computação é dividido em múltiplos microfluxos independentes. Com base no conteúdo de sua entrada (um bloco), cada microfluxo ajusta a quantidade de operação (nível de compressão) na análise da entrada de forma independente.
Como cada microfluxo é carregado na unidade de computação de um dispositivo (por exemplo, GPU) como uma rede neural individual, sua redução de operação é convertida diretamente em aceleração da latência.

Projeto de um controlador leve com reconhecimento de SLO, adaptável a orçamentos de computação limitados em dispositivos móveis/de borda (§5). Projetamos um controlador online leve, DEMUX, para ajustar a super-rede ARAC com base nos SLOs do usuário, com sobrecarga insignificante em dispositivos móveis e de borda. Dadas as opções personalizadas nos parâmetros de uma super-rede ARAC, o DEMUX seleciona adaptativamente o conjunto ideal de parâmetros e mantém alta precisão dentro dos SLOs do usuário.

Implementação da super-rede ARAC e avaliação de desempenho em diferentes plataformas de computação móvel/de borda e para várias aplicações de visão (§6, 7). Avaliamos de forma abrangente o desempenho da super-rede ARAC sob diversos aspectos e comprovamos sua eficácia em impulsionar o desempenho geral em aplicações relacionadas a CNN em dispositivos móveis/de borda. Destacamos nossos resultados de avaliação da seguinte forma:
- Supera as técnicas de última geração em inferência dinâmica e compressão de modelos em dispositivos móveis/de borda em até 67,9% (§7.2) e 1,23×(§7.4), respectivamente.
- Melhora o desempenho geral dos sistemas de detecção contínua de objetos de última geração na borda em até 7,7× (§7.6).
- Reduz a latência de ponta a ponta em quase 50% em um sistema de detecção de objetos 3D de última geração para dispositivos de realidade mista (§7.8).

## BACKGROUND E MOTIVAÇÃO