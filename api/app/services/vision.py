"""Componente de Visão (Applied Computer Vision).

Usa o YOLO11n treinado no RDD2022 (ver vision/treino_colab.ipynb) para
sugerir se a foto mostra dano na pista (buraco ou rachadura).

O produto tem uma única categoria automática, `dano_na_pista`. As quatro
classes do modelo servem só para o treino e para o relatório da disciplina.
A sugestão sempre passa pela confirmação do cidadão e não diz se a via é segura.

O modelo pode ser um .pt (YOLO via ultralytics) ou um .onnx (onnxruntime, sem
torch, bem mais leve). O .onnx publicado na imagem é o detector de buracos
treinado pelo Joseh (YOLO11n no RDD2022, classe única "buraco"); o pré e o
pós-processamento da inferência ONNX vêm do código de visão dele
(zeladoria-app/vision/inference.py).

Se o modelo não estiver disponível (variável VISION_MODEL_PATH vazia,
arquivo ausente ou pacote não instalado), a API continua funcionando sem
sugestão, e o cidadão escolhe a categoria.
"""

import io
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

ROTULO_DANO = "dano_na_pista"
ROTULO_SEM_DANO = "sem_deteccao"

# Classes do modelo (vision/rdd_para_yolo.py) que contam como dano na pista.
CLASSES_DANO = {"buraco", "rachadura_jacare", "rachadura_longitudinal", "rachadura_transversal"}
CONFIANCA_MINIMA = 0.25
# Limiar do detector ONNX (o mesmo do relatório de Visão).
CONFIANCA_ONNX = float(os.getenv("CV_CONFIDENCE", "0.35"))


@dataclass
class Deteccao:
    classe: str
    confianca: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Sugestao:
    label: str | None  # dano_na_pista | sem_deteccao | None (sem modelo)
    confidence: float | None  # 0 a 1
    model: str | None


SEM_MODELO = Sugestao(label=None, confidence=None, model=None)


def nome_modelo() -> str:
    return os.getenv("VISION_MODEL_NAME") or Path(os.getenv("VISION_MODEL_PATH", "")).name or "desconhecido"


def avaliar(deteccoes: list[Deteccao], modelo: str) -> Sugestao:
    """Dano na pista quando há ao menos uma detecção confiável; a confiança é a maior delas."""
    validas = [d for d in deteccoes if d.confianca >= CONFIANCA_MINIMA and d.classe in CLASSES_DANO]
    if not validas:
        return Sugestao(label=ROTULO_SEM_DANO, confidence=0.0, model=modelo)
    melhor = max(validas, key=lambda d: d.confianca)
    return Sugestao(label=ROTULO_DANO, confidence=round(melhor.confianca, 3), model=modelo)


class ModeloOnnx:
    """Sessão do onnxruntime para o detector de buraco (classe única, índice 0)."""

    def __init__(self, caminho: str):
        import onnxruntime as ort

        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = 1
        opcoes.inter_op_num_threads = 1
        opcoes.enable_mem_pattern = False
        self.sessao = ort.InferenceSession(caminho, sess_options=opcoes, providers=["CPUExecutionProvider"])


def _nms(caixas, notas, iou_max: float = 0.45, limite: int = 100) -> list[int]:
    import numpy as np

    ordem = np.argsort(notas)[::-1]
    escolhidas: list[int] = []
    while ordem.size and len(escolhidas) < limite:
        atual = int(ordem[0])
        escolhidas.append(atual)
        resto = ordem[1:]
        if not resto.size:
            break
        x1 = np.maximum(caixas[atual, 0], caixas[resto, 0])
        y1 = np.maximum(caixas[atual, 1], caixas[resto, 1])
        x2 = np.minimum(caixas[atual, 2], caixas[resto, 2])
        y2 = np.minimum(caixas[atual, 3], caixas[resto, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = max(0.0, caixas[atual, 2] - caixas[atual, 0]) * max(0.0, caixas[atual, 3] - caixas[atual, 1])
        area_r = np.maximum(0, caixas[resto, 2] - caixas[resto, 0]) * np.maximum(0, caixas[resto, 3] - caixas[resto, 1])
        iou = inter / np.maximum(area_a + area_r - inter, 1e-9)
        ordem = resto[iou <= iou_max]
    return escolhidas


def _detectar_onnx(modelo: ModeloOnnx, im) -> list[Deteccao]:
    import numpy as np
    from PIL import Image

    entrada = modelo.sessao.get_inputs()[0]
    lado = int(entrada.shape[-1]) if isinstance(entrada.shape[-1], int) else 640
    larg, alt = im.size
    escala = min(lado / larg, lado / alt)
    nova_larg, nova_alt = max(1, round(larg * escala)), max(1, round(alt * escala))
    redim = im.resize((nova_larg, nova_alt), Image.Resampling.BILINEAR)
    esq, topo = round((lado - nova_larg) / 2 - 0.1), round((lado - nova_alt) / 2 - 0.1)
    tela = Image.new("RGB", (lado, lado), (114, 114, 114))
    tela.paste(redim, (esq, topo))
    tensor = np.asarray(tela, dtype=np.float32).transpose(2, 0, 1)[None, ...] / 255.0
    saida = np.asarray(modelo.sessao.run(None, {entrada.name: tensor})[0]).squeeze()
    if saida.ndim != 2:
        raise ValueError("saída inesperada do detector ONNX")
    if saida.shape[0] <= 64 and saida.shape[0] < saida.shape[1]:
        saida = saida.T
    notas_classes = saida[:, 4:]
    mascara = (notas_classes.argmax(axis=1) == 0) & (notas_classes.max(axis=1) >= CONFIANCA_ONNX)
    linhas, notas = saida[mascara], notas_classes.max(axis=1)[mascara]
    if not len(linhas):
        return []
    caixas = np.empty((len(linhas), 4), dtype=np.float32)
    caixas[:, 0] = linhas[:, 0] - linhas[:, 2] / 2
    caixas[:, 1] = linhas[:, 1] - linhas[:, 3] / 2
    caixas[:, 2] = linhas[:, 0] + linhas[:, 2] / 2
    caixas[:, 3] = linhas[:, 1] + linhas[:, 3] / 2
    ok = (caixas[:, 2] > caixas[:, 0]) & (caixas[:, 3] > caixas[:, 1])
    caixas, notas = caixas[ok], notas[ok]
    resultado = []
    for i in _nms(caixas, notas):
        x1, y1, x2, y2 = ((caixas[i, 0] - esq) / escala, (caixas[i, 1] - topo) / escala,
                          (caixas[i, 2] - esq) / escala, (caixas[i, 3] - topo) / escala)
        resultado.append(Deteccao("buraco", float(notas[i]), max(x1, 0.0), max(y1, 0.0), min(x2, larg), min(y2, alt)))
    return resultado


@lru_cache(maxsize=1)
def _carregar_modelo():
    caminho = os.getenv("VISION_MODEL_PATH", "")
    if not caminho or not Path(caminho).is_file():
        log.warning("Modelo de visão não encontrado (%r); sem sugestão automática", caminho)
        return None
    if caminho.lower().endswith(".onnx"):
        try:
            return ModeloOnnx(caminho)
        except ImportError:
            log.warning("Pacote onnxruntime não instalado; sem sugestão automática")
            return None
    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning("Pacote ultralytics não instalado; sem sugestão automática")
        return None
    return YOLO(caminho)


def detectar(modelo, imagem: bytes) -> list[Deteccao]:
    from PIL import Image

    with Image.open(io.BytesIO(imagem)) as im:
        im = im.convert("RGB")
        if isinstance(modelo, ModeloOnnx):
            return _detectar_onnx(modelo, im)
        r = modelo.predict(im, imgsz=640, conf=CONFIANCA_MINIMA, verbose=False)[0]
    return [
        Deteccao(r.names[int(c)], float(p), *(float(v) for v in xyxy))
        for c, p, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)
    ]


def sugerir(imagem: bytes) -> Sugestao:
    modelo = _carregar_modelo()
    if modelo is None:
        return SEM_MODELO
    try:
        dets = detectar(modelo, imagem)
    except Exception:  # foto corrompida não pode derrubar o relato
        log.exception("Falha na inferência de visão")
        return SEM_MODELO
    return avaliar(dets, nome_modelo())
