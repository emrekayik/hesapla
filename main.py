import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
# Genel Pix2Text yerine, matematik için optimize edilmiş LatexOCR sınıfını kullanalım
from pix2text import LatexOCR
# matematiksel ifadeyi çözmek
import matplotlib.pyplot as plt
from sympy import Eq, symbols, Expr, lambdify
from sympy.plotting import plot
from latex2sympy2 import latex2sympy
from latex2sympy2 import latex2sympy, latex2latex
from utils import preprocess_image

# --- STREAMLIT UYGULAMASI ---

st.set_page_config(
    layout="wide",
    page_title="Hesapla",
    page_icon=":abacus:"
)
st.title("Hesapla :abacus:")
st.markdown("""
Bu uygulama, el yazısı matematiksel ifadeleri LaTeX formatına dönüştürür.
- Çizimlerinizi tanımak için Pix2Text'in LatexOCR modelini kullanır.
- Aşağıdaki tuvali kullanarak matematiksel ifadenizi çizin ve ardından "✅ İfadeyi Tanı ve LaTeX'e Çevir" butonuna tıklayın.
""")
st.info("Lütfen matematiksel ifadenizi aşağıdaki alana net bir şekilde çizin. Karmaşık ifadeler için daha büyük bir fırça boyutu deneyebilirsiniz.")

# cache kullandık model bir kere yüklenerek bellekte tutulacak
@st.cache_resource
def load_model():
    """LatexOCR modelini yükler ve bellekte tutar."""
    return LatexOCR()

# Model yüklenmesi
ocr_model = load_model()

# sütunlu görünüm
col1, col2 = st.columns(2)

with col1:
    st.subheader("Çizim Alanı")
    # Canvas ayarları
    stroke_width = st.slider("Fırça Kalınlığı", 1, 20, 4)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="white",
        background_color="black",
        update_streamlit=True,
        height=300,
        width=600,
        drawing_mode="freedraw",
        key="math_canvas"
    )

# Eğer tuvalde bir çizim varsa
if canvas_result.image_data is not None and canvas_result.image_data.any():
    if st.button("✅ İfadeyi Tanı ve LaTeX'e Çevir"):
        # 1. Görseli ön işleme fonksiyonundan geçir
        processed_image = preprocess_image(canvas_result.image_data)

        # Eğer canvas boş değilse devam et
        if processed_image:
            with col2:
                st.subheader("Sonuç")
                st.write("Modelin daha iyi anlaması için görsel işlendi:")
                st.image(processed_image, caption="🖼️ Kırpılmış ve İşlenmiş Görüntü")

                with st.spinner("🧠 Yapay zeka ifadeyi analiz ediyor..."):
                    try:
                        # 2. İşlenmiş görseli modele gönder.
                        results = ocr_model(processed_image)

                        # Modelin çıktısını işlemek için bir değişken tanımla
                        result_dict = None

                        # Modelin çıktısı liste mi yoksa doğrudan sözlük mü diye kontrol et
                        if isinstance(results, list) and results:
                            result_dict = results[0]
                        elif isinstance(results, dict):
                            result_dict = results

                        # Sonucun beklendiği gibi olup olmadığını kontrol et
                        if result_dict and 'text' in result_dict:
                            # Sözlükten LaTeX kodunu al
                            latex_code = result_dict['text']

                            st.success("🎉 Tanıma Başarılı!")
                            
                            # Tanınan LaTeX kodunu göster
                            st.subheader("Oluşturulan LaTeX Kodu")
                            st.code(latex_code, language="latex")

                            # LaTeX kodunun render edilmiş halini göster
                            st.subheader("Matematiksel Gösterim")
                            st.latex(latex_code)

                            try:
                                replaced_latex = latex2latex(latex_code)
                                st.subheader("Çözüm")
                                st.latex(replaced_latex)
                                st.write("Sembolik İfade:")
                                # LaTeX kodunu sembolik ifadeye çevir ve göster 
                                st.write(latex2sympy(latex_code))
                            except Exception as e:
                                st.error(f"❌ LaTeX kodu render edilirken bir hata oluştu: {e}")
                            
                            # plotting
                            try:
                                expr = latex2sympy(latex_code)

                                # Eşitlikse sol - sağ yapalım (denklem → fonksiyon formu)
                                if isinstance(expr, Eq):
                                    expr = expr.lhs - expr.rhs

                                # Tek değişkenli mi kontrol et (örneğin sadece x mi var)
                                if isinstance(expr, Expr) and len(expr.free_symbols) == 1:
                                    st.subheader("📈 Fonksiyon Grafiği")
                                    x = symbols('x')
                                    f = lambdify(x, expr, modules=["numpy"])
                                    x_vals = np.linspace(-10, 10, 400)
                                    y_vals = f(x_vals)
                                    fig, ax = plt.subplots()
                                    ax.plot(x_vals, y_vals, label=str(expr), color='blue', linewidth=2, alpha=0.7)
                                    ax.axhline(0, color='gray', linewidth=1)
                                    ax.axvline(0, color='gray', linewidth=1)
                                    ax.set_title("Fonksiyon Grafiği")
                                    ax.set_xlabel("x")
                                    ax.set_ylabel("f(x)")
                                    st.pyplot(fig)
                                else:
                                    st.warning("⚠️ Grafik sadece tek değişkenli fonksiyonlar için çizilebilir.")

                            except Exception as e:
                                st.error(f"❌ Grafik çizilirken bir hata oluştu: {e}")

                            # İndirme butonu
                            st.download_button(
                                label="📄 LaTeX çıktısını (.tex) indir",
                                data=latex_code,
                                file_name="math_output.tex",
                                mime="text/plain"
                            )
                        else:
                            st.warning("⚠️ Matematiksel bir ifade algılanamadı.")
                            # Hata ayıklama için modelin ne döndürdüğünü göster
                            st.write("Modelin ham çıktısı:", results)

                    except Exception as e:
                        st.error(f"❌ Bir hata oluştu: {e}")
        else:
            st.warning("Lütfen tanıma işlemi için bir şeyler çizin.")

st.markdown(
    """
    Bu uygulama Emre Kayık tarafından geliştirilmiştir. El yazısı matematiksel ifadeleri LaTeX formatına dönüştürmek için Pix2Text'in LatexOCR modelini kullanır.
    Kaynak kod: [GitHub](https://github.com/emrekayik/hesapla) 
    """
)