## Conceitos

### GRADIENTE

Para uma função de várias variáveis 

$$f(x1​,x2​,...,xn​)$$

O gradiente é o vetor formado pelas derivadas parciais da função em relação a cada variável:

$$∇f(x1​,x2​,...,xn​)=(frac{∂x1}{​∂f}​,frac{∂x2}​{∂f}​,...,/frac{∂xn}{​∂f}​)$$

Ilustração:
Imagine uma montanha representada por uma função f(x,y) — cada ponto (x,y) tem uma altura f(x,y).
Se você está em algum ponto e quer subir o mais rápido possível, o gradiente te diz em que direção andar.
O gradiente é a “direção de maior subida”.


### ÉPOCA

Uma época é uma passagem completa de todo o conjunto de dados de treino pelo modelo.

Ou seja, nesse caso, temos 2.000 imagens de gatos e cachorros e queremos treinar por 100 épocas, o modelo vai ver todas as 2.000 imagens 100 vezes, com possíveis variações de ordem e pequenas transformações, se você estiver usando data augmentation.

#### Em termos técnicos:

Durante o treinamento:

- O dataset é dividido em batches (lotes menores), por exemplo, 32 imagens por lote.
- O modelo faz uma passagem direta (forward) e uma retropropagação (backpropagation) para cada batch, atualizando os pesos.
- Quando todos os batches foram usados uma vez, isso completa 1 época.
- Em seguida, o modelo começa uma nova época, geralmente com os dados embaralhados (shuffle), para melhorar a generalização.

#### Por que usar várias épocas?

O modelo aprende gradualmente:
- Nas primeiras épocas, ele está “tentando entender” os padrões básicos.
- Depois de algumas épocas, começa a capturar os padrões importantes.
- Após muitas épocas, ele pode começar a decorar os dados de treino (isso é o overfitting).
- Por isso, você escolhe um número de épocas que permita ao modelo aprender bem sem memorizar os dados.

#### Exemplo prático (TensorFlow/Keras)
```python
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)
# epochs=10 → o modelo verá todo o dataset 10 vezes
# batch_size=32 → ele processa 32 exemplos por vez antes de atualizar os pesos
# validation_split=0.2 → usa 20% dos dados pra validar a performance após cada época
```

#### Analogia rápida:

Treinar um modelo por 1 época é como ler um livro uma vez. Você entende um pouco.
Treinar por 10 épocas é como reler o livro 10 vezes, você começa a fixar e compreender detalhe, mas se reler demais (100 épocas), você pode acabar decorando o texto sem realmente compreender o conteúdo.

### Overfitting

Overfitting ocorre quando um modelo aprende demais os detalhes e o “ruído” do conjunto de treino, a ponto de perder a capacidade de generalizar para dados novos (ou de validação/teste). Em outras palavras: <code>O modelo memoriza o treinamento em vez de aprender padrões gerais.</code>

Sintoma típico:
- Erro (ou perda) muito baixo no conjunto de treino ✅
- Erro alto no conjunto de validação/teste ❌

#### Como o Overfitting aparece em CNNs?

As CNNs são especialmente poderosas porque aprendem representações espaciais complexas — bordas, texturas, formas etc. Por terem muitos parâmetros (filtros, camadas convolucionais e totalmente conectadas), elas também são propensas a overfitting se não houver dados ou regularização suficiente. As causas principais são:
- Poucos dados de treino
- Se a rede vê poucas imagens, ela tende a memorizar exemplos específicos em vez de aprender padrões gerais.
- Modelo muito complexo
- Muitas camadas convolucionais, filtros grandes ou camadas densas com muitos neurônios.
- O poder de representação excede o necessário para o problema.
- Ausência de regularização
- Falta de técnicas como dropout, weight decay (L2 regularization) ou batch normalization.
- Falta de data augmentation
- Treino por tempo excessivo

CNNs precisam ver variações (rotação, zoom, brilho, inversão, etc.) para aprender robustez. Sem isso, memorizam as imagens exatamente como foram vistas.<br>
Mesmo com regularização, treinar por muitas épocas faz a rede se ajustar demais aos dados de treino. (Por isso se usa early stopping.)<br>
Distribuição diferente entre treino e validação.<br>
Quando o conjunto de validação vem de uma fonte ou condição diferente (ex: iluminação, fundo, tipo de câmera), o modelo se ajusta ao domínio do treino e falha fora dele.

### Install CUDA and Libraries

### Tensorflow CUDA setup

Para quem possuir GPU:

1. No terminal, digite:

```bash
nvidia-smi
```

Isso deve mostrar sua GPU, versão do driver e uso de memória. 
Se retornar um array vazio, será necessário instalar o driver NVIDIA.

2. Verifique compatibilidade CUDA/cuDNN com TensorFlow

TensorFlow precisa de versões específicas do CUDA e cuDNN.

Exemplo para TensorFlow 2.14+:

Componente	Versão recomendada
CUDA	12.2
cuDNN	8.9
Driver NVIDIA	≥ 530.30

Se você não tiver essas versões, TensorFlow não verá sua GPU.

3️⃣ Instale CUDA e cuDNN

Baixe o CUDA Toolkit compatível.
Baixe cuDNN compatível.

Extraia os arquivos e configure as variáveis de ambiente:

CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2
PATH=%CUDA_PATH%\bin;%PATH%

4️⃣ Reinstale TensorFlow

Depois de configurar CUDA e cuDNN corretamente:

pip uninstall tensorflow
pip install tensorflow

Não precisa instalar “tensorflow-gpu”, o tensorflow moderno já inclui suporte a GPU.

print(f'CUDA ativo: {torch.cuda.is_available()}')
print(f'Quantidade de GPUs disponíveis: {torch.cuda.device_count()}')
print(f'Nome da GPU: {torch.cuda.get_device_name(0)}')