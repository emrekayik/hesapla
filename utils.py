from PIL import Image, ImageOps
import numpy as np


# --- GÖRSEL ÖN İŞLEME FONKSİYONU ---
# Bu fonksiyon, canvas'tan gelen görseli OCR modeli için optimize eder.
def preprocess_image(image_data: np.ndarray) -> Image.Image | None:
    """
    Canvas'tan gelen RGBA görüntüsünü işler:
    1. Beyaz bir arka plan üzerine yapıştırarak RGB'ye dönüştürür.
    2. Çizimin etrafındaki boşlukları kırpar (bounding box).
    3. Karakterlerin kenarlara yapışmaması için etrafına boşluk (padding) ekler.
    """
    # RGBA'dan PIL Image nesnesine dönüştür
    img = Image.fromarray(image_data.astype("uint8"), "RGBA")

    # Tamamen beyaz, aynı boyutlarda bir arka plan oluştur
    background = Image.new("RGB", img.size, (255, 255, 255))

    # Çizimi (img), alfa kanalını maske olarak kullanarak beyaz arka planın üzerine yapıştır
    background.paste(img, (0, 0), img)

    # Çizimin yapıldığı alanı bulmak için görseli ters çevirip sınırlayıcı kutuyu (bounding box) al
    # Gri tonlamaya çevirip tersini almak, siyah piksellerin (çizimin) beyaz olmasını sağlar
    inverted_img = ImageOps.invert(background.convert("L"))
    bbox = inverted_img.getbbox()

    # Eğer canvas boş değilse (bir çizim varsa)
    if bbox:
        # 5. Sınırlayıcı kutuya göre görseli kırp
        cropped_img = background.crop(bbox)

        # 6. Kırpılmış görselin etrafına 20 piksel beyaz boşluk ekle
        padded_img = ImageOps.expand(cropped_img, border=20, fill='white')
        return padded_img
    
    # Canvas boşsa None döndür
    return None