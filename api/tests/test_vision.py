from app.services import vision
from app.services.vision import Deteccao, avaliar


def caixa(classe, conf=0.9):
    return Deteccao(classe, conf, 0, 0, 100, 100)


def test_sem_deteccao():
    r = avaliar([], "best.pt")
    assert (r.label, r.confidence, r.model) == ("sem_deteccao", 0.0, "best.pt")


def test_buraco_e_rachaduras_contam_como_dano_na_pista():
    for classe in ("buraco", "rachadura_jacare", "rachadura_longitudinal", "rachadura_transversal"):
        assert avaliar([caixa(classe)], "m").label == "dano_na_pista"


def test_confianca_e_a_maior_deteccao_valida():
    dets = [caixa("rachadura_transversal", 0.4), caixa("buraco", 0.77), caixa("classe_estranha", 0.99)]
    assert avaliar(dets, "m").confidence == 0.77


def test_ignora_baixa_confianca():
    assert avaliar([caixa("buraco", conf=0.1)], "m").label == "sem_deteccao"


def test_sem_modelo_nao_sugere(monkeypatch):
    monkeypatch.setattr(vision, "_carregar_modelo", lambda: None)
    assert vision.sugerir(b"qualquer") == vision.SEM_MODELO


def test_falha_na_inferencia_nao_derruba(monkeypatch):
    monkeypatch.setattr(vision, "_carregar_modelo", lambda: object())

    def quebra(*_):
        raise ValueError("foto corrompida")

    monkeypatch.setattr(vision, "detectar", quebra)
    assert vision.sugerir(b"x") == vision.SEM_MODELO


def test_usa_resultado_do_modelo(monkeypatch):
    monkeypatch.setattr(vision, "_carregar_modelo", lambda: object())
    monkeypatch.setattr(vision, "detectar", lambda m, img: [caixa("buraco", 0.66)])
    monkeypatch.setenv("VISION_MODEL_NAME", "yolo11n-rdd2022")
    r = vision.sugerir(b"x")
    assert (r.label, r.confidence, r.model) == ("dano_na_pista", 0.66, "yolo11n-rdd2022")


# --- detector ONNX (modelo de buracos incluído em api/modelos) ---
import io
from pathlib import Path

import pytest

MODELO_ONNX = Path(__file__).resolve().parents[1] / "modelos" / "best.onnx"


@pytest.fixture
def com_onnx(monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("VISION_MODEL_PATH", str(MODELO_ONNX))
    monkeypatch.setenv("VISION_MODEL_NAME", "yolo11n-rdd2022.onnx")
    vision._carregar_modelo.cache_clear()
    yield
    vision._carregar_modelo.cache_clear()


def _png(cor, tamanho=(320, 240)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", tamanho, cor).save(buf, "PNG")
    return buf.getvalue()


def test_onnx_carrega_e_nao_ve_dano_em_imagem_lisa(com_onnx):
    r = vision.sugerir(_png((128, 128, 128)))
    assert r.label == "sem_deteccao" and r.confidence == 0.0
    assert r.model == "yolo11n-rdd2022.onnx"


def test_onnx_devolve_sugestao_valida_para_foto_real(com_onnx):
    foto = Path(__file__).resolve().parents[2] / "docs" / "imagens" / "buraco_bairro_piloto.jpg"
    r = vision.sugerir(foto.read_bytes())
    assert r.label in {"dano_na_pista", "sem_deteccao"}
    assert r.model == "yolo11n-rdd2022.onnx"
    assert r.confidence is not None and 0.0 <= r.confidence <= 1.0


def test_onnx_foto_corrompida_nao_derruba(com_onnx):
    assert vision.sugerir(b"isto nao e uma imagem") == vision.SEM_MODELO
