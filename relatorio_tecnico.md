# Relatório Técnico — Pipeline de Contagem de Parafusos por Visão Computacional

## Sumário

1. [Visão Geral do Pipeline](#1-visão-geral-do-pipeline)
2. [Etapa 1 — Conversão para Escala de Cinza](#2-etapa-1--conversão-para-escala-de-cinza)
3. [Etapa 2 — Suavização com Filtro Gaussiano](#3-etapa-2--suavização-com-filtro-gaussiano)
4. [Etapa 3 — Binarização Adaptativa](#4-etapa-3--binarização-adaptativa)
5. [Etapa 4 — Operações Morfológicas](#5-etapa-4--operações-morfológicas)
6. [Etapa 5 — Algoritmo Watershed](#6-etapa-5--algoritmo-watershed)
7. [Etapa 6 — Remoção de Bordas](#7-etapa-6--remoção-de-bordas)
8. [Etapa 7 — Detecção de Contornos](#8-etapa-7--detecção-de-contornos)
9. [Etapa 8 — Filtros por Métricas de Forma](#9-etapa-8--filtros-por-métricas-de-forma)
   - [Área](#91-área-area_min-e-area_max)
   - [Solidez](#92-solidez-solidez_min)
   - [Aspecto](#93-razão-de-aspecto-aspecto_min-e-aspecto_max)
   - [Extent](#94-extent-extent_min)
10. [Tabela de Referência Rápida dos Parâmetros](#10-tabela-de-referência-rápida-dos-parâmetros)
11. [Guia de Diagnóstico — O que fazer quando a contagem erra](#11-guia-de-diagnóstico--o-que-fazer-quando-a-contagem-erra)

---

## 1. Visão Geral do Pipeline

O pipeline transforma uma fotografia colorida de parafusos numa contagem numérica por meio de oito etapas sequenciais de processamento de imagem. Cada etapa prepara a imagem para a seguinte, refinando progressivamente a representação até que objetos individuais possam ser identificados e classificados.

```
Imagem Original (BGR)
        │
        ▼
   Escala de Cinza
        │
        ▼
   Blur Gaussiano          ← reduz ruído antes da limiarização
        │
        ▼
   Binarização Adaptativa  ← separa objeto do fundo
        │
        ▼
   Morfologia (Close→Open) ← fecha buracos, remove ruído residual
        │
        ▼
   Watershed (opcional)    ← separa parafusos que se tocam
        │
        ▼
   Remoção de Bordas        ← descarta artefatos nas margens
        │
        ▼
   Extração de Contornos
        │
        ▼
   Filtros de Forma         ← área, solidez, aspecto, extent
        │
        ▼
   Contagem Final + Visualização
```

---

## 2. Etapa 1 — Conversão para Escala de Cinza

### O que faz

Colapsa os três canais de cor (R, G, B) num único canal de intensidade luminosa. A fórmula padrão do OpenCV pondera os canais de acordo com a percepção visual humana:

```
L = 0.299·R + 0.587·G + 0.114·B
```

### Por que é importante

Todos os algoritmos subsequentes (limiarização, morfologia, transformada de distância) operam em imagens de um único canal. Trabalhar em escala de cinza reduz o volume de dados em 3× e simplifica os cálculos sem perder as informações estruturais relevantes — contornos, texturas e contrastes de brilho — que permitem distinguir parafusos do fundo.

### Como ajustar

Não há parâmetro configurável aqui. Se a imagem original tiver um canal de cor com muito mais contraste entre parafuso e fundo (por exemplo, em fotografia com iluminação colorida), pode valer a pena experimentar usar apenas esse canal isolado em vez da conversão padrão (`cv2.split(imagem_bgr)` retorna B, G, R separados).

---

## 3. Etapa 2 — Suavização com Filtro Gaussiano

### Parâmetro

```python
BLUR_KERNEL = (5, 5)
```

### O que faz

Aplica uma convolução com um kernel gaussiano bidimensional sobre a imagem em escala de cinza. Cada pixel recebe um valor médio ponderado dos pixels vizinhos, onde vizinhos mais próximos têm peso maior. O resultado é uma imagem "borrada" que elimina variações abruptas de intensidade causadas por ruído de sensor, grão fotográfico e pequenas imperfeições de superfície.

### Por que é importante

Sem suavização, pequenas variações de brilho dentro do corpo de um parafuso geram bordas falsas durante a binarização. Isso fragmenta um único parafuso em dezenas de regiões desconexas, inviabilizando a contagem. O blur pré-processamento é a principal defesa contra esse problema.

### Como ajustar

O kernel deve ser sempre uma tupla de dois valores **ímpares** e **iguais** (o OpenCV exige dimensões ímpares para centrar corretamente o kernel).

| Valor do Kernel | Efeito | Quando usar |
|---|---|---|
| `(3, 3)` | Suavização leve | Imagens nítidas com pouco ruído |
| `(5, 5)` | Suavização moderada | Padrão, bom para a maioria dos casos |
| `(7, 7)` | Suavização agressiva | Imagens com muito ruído ou granularidade |
| `(11, 11)` | Borramento forte | Texturas muito complexas no fundo |

**Regra prática:** aumente o kernel se a binarização resultar em muitos fragmentos pequenos dentro dos parafusos. Diminua se detalhes importantes de borda estiverem sendo perdidos, fazendo parafusos fundirem com o fundo.

---

## 4. Etapa 3 — Binarização Adaptativa

### Parâmetros internos (na chamada `cv2.adaptiveThreshold`)

```python
blockSize = 51
C         = 4
method    = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
type      = cv2.THRESH_BINARY_INV
```

### O que faz

Converte a imagem em escala de cinza numa imagem binária (preto e branco puro). Para cada pixel, calcula um limiar **local** baseado nos pixels vizinhos dentro de uma janela de tamanho `blockSize × blockSize`. O limiar é a média gaussiana ponderada da vizinhança menos a constante `C`. Pixels mais escuros que o limiar tornam-se brancos (valor 255); os mais claros tornam-se pretos (valor 0). O `_INV` inverte esse resultado — objetos escuros sobre fundo claro ficam brancos na imagem binária.

### Por que a limiarização adaptativa é superior à global

Fotografias de parafusos raramente têm iluminação uniforme. Uma parte da imagem pode estar mais iluminada, outra em sombra. Um limiar global único (como o de Otsu) funcionaria bem na região bem iluminada mas falha nas sombras — parafusos escurecidos se fundem com o fundo. A limiarização adaptativa resolve isso calculando um limiar diferente para cada região, compensando as variações de iluminação automaticamente.

### Como ajustar `blockSize`

`blockSize` define o tamanho da vizinhança usada para calcular o limiar local. Deve ser um número **ímpar** e maior que o tamanho dos detalhes de textura que você quer ignorar.

| Valor | Efeito | Quando usar |
|---|---|---|
| `11–21` | Limiar muito local, captura micro-texturas | Parafusos pequenos, detalhes finos |
| `31–51` | Limiar moderadamente local | Padrão recomendado |
| `71–101` | Limiar abrangente, ignora variações locais | Iluminação com gradiente suave e uniforme |

**Regra prática:** se parafusos com fundo muito claro estão desaparecendo, aumente `blockSize`. Se o fundo está sendo detectado como objeto, diminua.

### Como ajustar `C`

`C` é subtraído do valor médio calculado antes de compará-lo com o pixel. Valores maiores de `C` tornam o limiar mais alto, resultando em menos pixels classificados como objeto (imagem binária mais escura = menos falsos positivos de fundo). Valores menores ou negativos incluem mais pixels como objeto.

| Valor | Efeito |
|---|---|
| `C = 1–3` | Limiar baixo — captura mais da superfície dos objetos |
| `C = 4–6` | Equilíbrio padrão |
| `C = 8–12` | Limiar alto — inclui apenas as bordas e partes mais escuras |

---

## 5. Etapa 4 — Operações Morfológicas

### Parâmetros

```python
CLOSE_KERNEL = (9, 9)   # tamanho do kernel da operação Close
CLOSE_ITER   = 1        # número de iterações do Close
MORPH_KERNEL = (3, 3)   # tamanho do kernel da operação Open
MORPH_ITER   = 2        # número de iterações do Open
```

### Conceito fundamental: erosão e dilatação

Todas as operações morfológicas são combinações de duas primitivas:

- **Dilatação:** expande regiões brancas — pequenos buracos somem, bordas crescem
- **Erosão:** encolhe regiões brancas — pequenas manchas somem, bordas recuam

### Operação Close (Fechamento)

**Sequência:** Dilatação → Erosão

O fechamento preenche **buracos internos** e **lacunas pequenas** dentro dos objetos brancos sem alterar significativamente o contorno externo. No contexto dos parafusos, a cabeça e o corpo do parafuso frequentemente aparecem como regiões com pequenos buracos escuros (reflexos, texturas de rosca) após a binarização. O Close une essas partes num blob contínuo.

```python
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL)
fechada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_close, iterations=CLOSE_ITER)
```

**Como ajustar `CLOSE_KERNEL`:**

| Valor | Efeito |
|---|---|
| `(5, 5)` | Fecha lacunas pequenas (< 5 px) |
| `(9, 9)` | Fecha lacunas médias, padrão para parafusos comuns |
| `(15, 15)` | Fecha lacunas grandes — risco de fundir objetos próximos |
| `(21, 21)` | Fusão agressiva — usar só se parafusos estiverem muito fragmentados |

**Como ajustar `CLOSE_ITER`:**  
Cada iteração aplica o Close uma vez adicionalmente. `CLOSE_ITER = 2` é equivalente a fazer um Close com kernel maior. Prefira aumentar o kernel ao invés das iterações para ter controle mais preciso.

**Como ajustar `MORPH_KERNEL` (Open):**

A operação **Open** (Erosão → Dilatação) faz o oposto: remove **manchas pequenas e ruídos** isolados sem alterar objetos maiores.

| Valor | Efeito |
|---|---|
| `(2, 2)` | Remove apenas pixel único de ruído |
| `(3, 3)` | Remove manchas de até ~3 px — padrão |
| `(5, 5)` | Remove manchas maiores — risco de corroer bordas de objetos pequenos |

**Como ajustar `MORPH_ITER` (iterações do Open):**  
Aumentar iterações do Open erode progressivamente os objetos. Com `MORPH_ITER = 3` ou `4`, parafusos pequenos podem desaparecer completamente. Mantenha em 1–2 para a maioria dos casos.

**Ordem importa:** sempre aplique Close **antes** do Open. O Close preenche buracos primeiro (consolida o objeto); o Open depois remove ruídos externos sem desfazer o que o Close construiu.

---

## 6. Etapa 5 — Algoritmo Watershed

### Parâmetros

```python
USAR_WATERSHED        = True   # ativa/desativa o algoritmo
WATERSHED_DIST_LIMIAR = 0.3    # fração do máximo da transformada de distância
```

### O problema que o Watershed resolve

Quando dois ou mais parafusos estão em contato ou sobrepostos, operações morfológicas os fundem num único blob. Sem separação, todo o grupo seria contado como um único parafuso. O Watershed é o algoritmo clássico para separar objetos que se tocam.

### Como funciona (passo a passo)

1. **Transformada de Distância:** para cada pixel branco na máscara binária, calcula a distância até o pixel preto mais próximo. O resultado é uma "topografia" — pixels no centro de um objeto estão no "topo" da montanha; pixels na borda estão na "base".

2. **Definição do foreground seguro:** aplica um limiar sobre a transformada de distância. Apenas pixels com distância maior que `WATERSHED_DIST_LIMIAR × distância_máxima` são considerados "centro com certeza" de um objeto. Isso identifica os núcleos de cada parafuso separado.

3. **Definição do background seguro:** dilata levemente a máscara para garantir uma área ao redor dos objetos que é definitivamente fundo.

4. **Zona desconhecida:** pixels entre o foreground e background seguros — a região de fronteira ambígua onde a separação acontece.

5. **Componentes conectados:** rotula cada região de foreground seguro com um ID único — esses são os "marcadores" iniciais.

6. **Inundação:** o Watershed "inunda" a topografia a partir de cada marcador. Quando duas frentes de inundação se encontram, uma linha divisória é traçada — essa é a separação entre parafusos.

### Como ajustar `WATERSHED_DIST_LIMIAR`

Este é o parâmetro mais crítico do Watershed.

| Valor | Efeito | Quando usar |
|---|---|---|
| `0.1 – 0.2` | Limiar baixo — mais pixels são "centro seguro" | Parafusos pequenos, pode sobre-segmentar |
| `0.3 – 0.4` | Equilíbrio padrão — bom para parafusos médios | Caso geral |
| `0.5 – 0.6` | Limiar conservador — apenas núcleos centrais | Parafusos grandes com muito espaço interno |
| `0.7+` | Muito conservador — risco de não detectar núcleos | Evitar salvo casos extremos |

**Regra prática:**
- Se objetos que **não deveriam ser separados** estão sendo divididos (falsa segmentação), **aumente** o limiar (ex.: `0.3 → 0.5`).
- Se parafusos que **deveriam ser separados** continuam fundidos, **diminua** o limiar (ex.: `0.3 → 0.2`).

### Quando desativar o Watershed (`USAR_WATERSHED = False`)

- Imagens com parafusos bem separados entre si — o Watershed é computacionalmente mais caro e desnecessário.
- Imagens com muito ruído onde o Watershed cria centenas de micro-segmentos.
- Fase inicial de ajuste de parâmetros — desative para entender primeiro o comportamento base do pipeline.

---

## 7. Etapa 6 — Remoção de Bordas

### Parâmetro

```python
MARGEM = 30   # largura em pixels da faixa de borda a zerar
```

### O que faz

Após o processamento morfológico, as quatro bordas da imagem são zeradas (pixels definidos como preto): os primeiros e últimos `MARGEM` pixels em cada direção. Isso elimina objetos parcialmente visíveis que foram cortados pelo enquadramento da fotografia.

### Por que é importante

Um parafuso cortado pela borda da imagem teria uma forma incompleta — ele passaria pelos filtros de área (é menor que o esperado) ou seria rejeitado pelos filtros de forma (aspecto e solidez distorcidos pela parte faltante). Mais grave: sem a remoção de bordas, artefatos de processamento que ocorrem tipicamente nas margens (efeitos de borda da convolução, reflexos de flash) entrariam nos filtros e gerariam falsos positivos.

### Como ajustar `MARGEM`

| Valor | Efeito | Quando usar |
|---|---|---|
| `10 – 15` | Margem mínima | Imagens com parafusos muito próximos às bordas |
| `20 – 30` | Padrão, elimina a maioria dos artefatos de borda | Caso geral |
| `50+` | Margem larga | Fotografias com reflectos ou bordas escuras acentuadas |

**Atenção:** se parafusos válidos estão próximos às bordas da imagem, uma margem muito grande vai descartá-los. Neste caso, o correto é ajustar a câmera/enquadramento para que os parafusos fiquem centralizados, ou reduzir a margem.

---

## 8. Etapa 7 — Detecção de Contornos

### O que faz

`cv2.findContours` percorre a imagem binária e identifica os limites externos de cada região branca conectada, retornando uma lista de contornos — cada contorno é um conjunto de pontos (x, y) que formam o polígono de borda do objeto.

Os parâmetros `cv2.RETR_EXTERNAL` e `cv2.CHAIN_APPROX_SIMPLE` significam:
- **RETR_EXTERNAL:** retorna apenas o contorno externo de cada blob (ignora buracos internos).
- **CHAIN_APPROX_SIMPLE:** comprime segmentos horizontais, verticais e diagonais, armazenando apenas os pontos extremos — economiza memória e tempo.

### Por que é importante

Os contornos são a base para calcular todas as métricas de forma usadas nos filtros da próxima etapa. Sem eles, seria impossível medir área, solidez ou aspecto de cada candidato a parafuso.

---

## 9. Etapa 8 — Filtros por Métricas de Forma

Esta é a etapa de classificação: cada contorno detectado é avaliado segundo quatro critérios independentes. Um contorno precisa **passar em todos** para ser contado como parafuso válido.

### 9.1 Área (`AREA_MIN` e `AREA_MAX`)

```python
AREA_MIN = 200      # pixels²
AREA_MAX = 50_000   # pixels²
```

**O que é:**  
A área é o número de pixels dentro do contorno, calculada por `cv2.contourArea()`. É a métrica mais simples e geralmente a mais discriminativa.

**Por que é importante:**  
- `AREA_MIN` elimina ruídos, sujeira, reflexos minúsculos e fragmentos morfológicos que sobrevivem ao processamento.
- `AREA_MAX` descarta regiões enormes que claramente não são parafusos individuais — sombras grandes, reflexos de iluminação que cobrem metade da imagem, junções de múltiplos parafusos que o Watershed não separou.

**Como ajustar:**

O ajuste correto de `AREA_MIN` e `AREA_MAX` depende diretamente da **resolução da câmera** e da **distância de captura**. Parafusos maiores ou fotografados de mais perto ocuparão mais pixels; parafusos menores ou de longe, menos.

**Procedimento recomendado:**
1. Ative `MODO_DEBUG = True` e processe uma imagem representativa.
2. Observe a visualização final — contornos rejeitados mostram o motivo da rejeição (ex.: `area=150`).
3. Anote a faixa de área dos parafusos válidos e dos falsos positivos.
4. Defina `AREA_MIN` ligeiramente abaixo do menor parafuso válido e `AREA_MAX` ligeiramente acima do maior.

| Cenário | AREA_MIN recomendado | AREA_MAX recomendado |
|---|---|---|
| Parafusos pequenos, câmera próxima | 100 – 300 | 5.000 – 15.000 |
| Parafusos médios, distância padrão | 200 – 500 | 20.000 – 60.000 |
| Parafusos grandes ou câmera afastada | 500 – 2.000 | 80.000 – 200.000 |

### 9.2 Solidez (`SOLIDEZ_MIN`)

```python
SOLIDEZ_MIN = 0.35
```

**O que é:**  
A solidez (também chamada de convexidade) mede quão "cheio" é um objeto em relação à sua **envoltória convexa** (o menor polígono convexo que envolve o contorno). A fórmula é:

```
Solidez = Área do Contorno / Área da Envoltória Convexa
```

Um círculo perfeito ou um quadrado têm solidez = 1.0. Uma estrela ou um "C" aberto têm solidez próxima de 0.

**Por que é importante:**  
Parafusos vistos de cima ou de lado são objetos razoavelmente compactos — a cabeça hexagonal, cilíndrica ou Phillips tem solidez tipicamente entre 0.7 e 1.0. Artefatos de ruído, sombras e reflexos tendem a ter formas irregulares e recortadas, com solidez muito baixa (< 0.4).

**Como ajustar:**

| Valor de `SOLIDEZ_MIN` | Efeito |
|---|---|
| `0.10 – 0.20` | Filtro muito permissivo — aceita formas bastante irregulares |
| `0.35 – 0.50` | Padrão — elimina a maioria dos artefatos sem descartar parafusos inclinados |
| `0.60 – 0.75` | Filtro exigente — aceita apenas formas compactas e simétricas |
| `0.80+` | Muito restritivo — parafusos parcialmente sobrepostos ou inclinados serão rejeitados |

**Atenção:** parafusos vistos de lado (perfil) podem ter solidez mais baixa que os vistos de cima. Se suas imagens contêm parafusos em diferentes orientações, mantenha `SOLIDEZ_MIN` em torno de 0.35.

### 9.3 Razão de Aspecto (`ASPECTO_MIN` e `ASPECTO_MAX`)

```python
ASPECTO_MIN = 0.10
ASPECTO_MAX = 10.0
```

**O que é:**  
A razão de aspecto é a proporção entre a largura e a altura do **retângulo alinhado com os eixos** (bounding box) que envolve o contorno:

```
Aspecto = Largura / Altura
```

Um objeto quadrado tem aspecto = 1.0. Um objeto muito largo tem aspecto >> 1. Um objeto muito alto tem aspecto << 1.

**Por que é importante:**  
Parafusos vistos de cima (cabeça circular) têm aspecto próximo de 1.0 (±20%). Parafusos vistos de lado (corpo cilíndrico) podem ser bem mais longos que largos, com aspecto de 3 a 8. O filtro elimina objetos extremamente alongados (linhas, fios, bordas da imagem) ou extremamente achatados.

**Como ajustar:**

Os limites atuais `(0.10, 10.0)` são muito permissivos e funcionam como "última defesa" contra extremos absurdos. Para um controle mais preciso:

| Tipo de parafuso | ASPECTO_MIN sugerido | ASPECTO_MAX sugerido |
|---|---|---|
| Somente cabeça (vista de cima) | 0.6 | 1.8 |
| Cabeça + corpo (vista mista) | 0.3 | 5.0 |
| Qualquer orientação | 0.10 | 10.0 (padrão) |

**Cuidado:** estreitar demais a faixa de aspecto vai rejeitar parafusos levemente rotacionados ou cujas caixas delimitadoras incluam uma pequena sombra lateral.

### 9.4 Extent (`EXTENT_MIN`)

```python
EXTENT_MIN = 0.15
```

**O que é:**  
O extent mede a proporção entre a área real do contorno e a área do seu retângulo delimitador (bounding box):

```
Extent = Área do Contorno / (Largura × Altura da Bounding Box)
```

Um quadrado perfeito alinhado com os eixos tem extent = 1.0. Um círculo perfeito tem extent ≈ 0.785. Uma forma em "L" ou um triângulo têm extent baixo porque a bounding box contém muito espaço vazio.

**Por que é importante:**  
O extent complementa a solidez: enquanto a solidez compara o objeto com sua forma convexa ideal, o extent compara com o retângulo que o contém. Objetos muito espalhados ou com muitos "braços" (estrelas, silhuetas de parafusos fragmentados, L-shapes) têm extent baixo. Um parafuso real, mesmo irregular, normalmente preenche pelo menos 15–30% do seu bounding box.

**Como ajustar:**

| Valor de `EXTENT_MIN` | Efeito |
|---|---|
| `0.05 – 0.10` | Quase sem filtro — aceita formas muito esparsas |
| `0.15 – 0.20` | Padrão — elimina apenas formas extremamente dispersas |
| `0.30 – 0.40` | Moderado — exige que o objeto preencha bem sua bounding box |
| `0.50+` | Restritivo — favorece objetos compactos e simétricos |

**Relação entre Solidez e Extent:**  
- Alta solidez + baixo extent: objeto compacto mas mal alinhado com os eixos (ex.: parafuso rotacionado 45°).
- Baixa solidez + alto extent: objeto com recortes internos mas dentro de uma bounding box estreita.
- Ambos baixos: fragmento irregular — quase certamente não é um parafuso.

---

## 10. Tabela de Referência Rápida dos Parâmetros

| Parâmetro | Valor Atual | Faixa Típica | Efeito de Aumentar | Efeito de Diminuir |
|---|---|---|---|---|
| `BLUR_KERNEL` | `(5,5)` | `(3,3)–(11,11)` | Mais suave, perde bordas | Mais ruído na binarização |
| `CLOSE_KERNEL` | `(9,9)` | `(5,5)–(21,21)` | Fecha lacunas maiores, pode fundir objetos | Lacunas internas persistem |
| `CLOSE_ITER` | `1` | `1–3` | Fechamento mais agressivo | Menos fechamento |
| `MORPH_KERNEL` | `(3,3)` | `(2,2)–(7,7)` | Remove manchas maiores, corrói objetos pequenos | Mantém mais ruídos |
| `MORPH_ITER` | `2` | `1–4` | Erosão progressiva, objetos encolhem | Menos limpeza de ruído |
| `MARGEM` | `30` | `10–80` | Descarta mais da borda | Artefatos de borda entram |
| `WATERSHED_DIST_LIMIAR` | `0.3` | `0.1–0.7` | Menos separações (núcleos menores) | Mais separações (pode sobre-segmentar) |
| `AREA_MIN` | `200` | `50–2.000` | Elimina mais pequenos objetos | Aceita mais ruídos |
| `AREA_MAX` | `50.000` | `5.000–200.000` | Descarta objetos menores | Aceita aglomerados grandes |
| `SOLIDEZ_MIN` | `0.35` | `0.1–0.8` | Exige formas mais convexas | Aceita formas irregulares |
| `ASPECTO_MIN` | `0.10` | `0.1–0.6` | Exige objetos menos estreitos | Aceita objetos muito finos |
| `ASPECTO_MAX` | `10.0` | `2.0–10.0` | Aceita mais formas alongadas | Rejeita objetos muito compridos |
| `EXTENT_MIN` | `0.15` | `0.05–0.5` | Exige objetos mais compactos | Aceita formas muito esparsas |

---

## 11. Guia de Diagnóstico — O que fazer quando a contagem erra

### Cenário A: Parafusos válidos sendo rejeitados

Ative `MODO_DEBUG = True` e observe o motivo escrito em vermelho sobre os contornos rejeitados.

| Motivo exibido | Parâmetro a ajustar | Direção |
|---|---|---|
| `area=150` (muito pequeno) | `AREA_MIN` | Diminuir |
| `area=55000` (muito grande) | `AREA_MAX` | Aumentar |
| `solidez=0.28` | `SOLIDEZ_MIN` | Diminuir |
| `aspecto=0.08` | `ASPECTO_MIN` | Diminuir |
| `aspecto=11.2` | `ASPECTO_MAX` | Aumentar |
| `extent=0.12` | `EXTENT_MIN` | Diminuir |

### Cenário B: Ruídos e falsos positivos sendo contados

| Problema observado | Parâmetro a ajustar | Direção |
|---|---|---|
| Muitos fragmentos minúsculos | `AREA_MIN` | Aumentar |
| Sombras grandes sendo detectadas | `AREA_MAX` + `SOLIDEZ_MIN` | Diminuir `AREA_MAX` / Aumentar `SOLIDEZ_MIN` |
| Fundo texturizado gerando detecções | `BLUR_KERNEL` + `MORPH_KERNEL` | Aumentar ambos |
| Objetos nas bordas sendo contados | `MARGEM` | Aumentar |

### Cenário C: Parafusos agrupados sendo contados como um só

| Problema | Solução |
|---|---|
| Watershed desativado | Ativar `USAR_WATERSHED = True` |
| Watershed ativo mas não separando | Diminuir `WATERSHED_DIST_LIMIAR` (ex.: `0.3 → 0.2`) |
| `CLOSE_KERNEL` muito grande | Reduzir (ex.: `(9,9) → (5,5)`) — o Close pode estar fundindo objetos |

### Cenário D: Parafuso único sendo dividido em dois

| Problema | Solução |
|---|---|
| Watershed segmentando demais | Aumentar `WATERSHED_DIST_LIMIAR` (ex.: `0.3 → 0.5`) |
| Objeto tem buraco interno grande | Aumentar `CLOSE_KERNEL` para fechar o buraco antes do Watershed |
| Parafuso tem formato muito irregular | Ajustar `BLUR_KERNEL` para suavizar mais antes da binarização |

---

*Relatório gerado para o pipeline `contar_parafuso_batch.ipynb` — Desafio de Modelagem Preditiva.*
