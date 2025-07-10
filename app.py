import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch

st.set_page_config(page_title="MIDAR AI - Demo", layout="centered")
st.title("🧠 MIDAR AI - Demo de Segmentación Médica")

uploaded_file = st.file_uploader("Sube una imagen médica (.nii.gz)", type=["nii.gz"])

if uploaded_file:
    st.success("Imagen cargada correctamente ✅")

    # Guardar archivo temporal
    with open("temp.nii.gz", "wb") as f:
        f.write(uploaded_file.read())

    # Cargar con nibabel
    img = nib.load("temp.nii.gz")
    data = img.get_fdata()
    slice_idx = data.shape[2] // 2
    slice_data = data[:, :, slice_idx]
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))

    # Mostrar imagen original
    st.subheader("🖼 Imagen Original")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(slice_data, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

    # Crear modelo demo y aplicar
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    model.eval()
    x = torch.tensor(slice_data[None, None, :, :], dtype=torch.float32)
    with torch.no_grad():
        mask = model(x).sigmoid().numpy()[0, 0]

    # Mostrar segmentación
    st.subheader("🔍 Segmentación (Demo IA)")
    fig, ax = plt.subplots()
    ax.imshow(slice_data, cmap="gray")
    ax.contour(mask, levels=[0.5], colors="r")
    ax.axis("off")
    st.pyplot(fig)
