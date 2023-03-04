import ast
from io import BytesIO
import os
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv


load_dotenv()


def get_results(image):
    """Get Recommendations from the model"""
    buf = BytesIO()
    image.save(buf, "jpeg")
    image_bytes = buf.getvalue()
    op = requests.post(
        "http://0.0.0.0:3000/predict",
        data=image_bytes,
        headers={"content-type": "image/jpeg"},
    )
    return ast.literal_eval(
        op.content.decode("utf-8")
    )


@st.cache_data
def get_pandas_dataframe(path):
    """Read dataframe."""
    df = pd.read_csv(path)
    df.product_ids = df.product_page_links.apply(
        lambda x: x.split(".")[-2]
    )
    return df


def get_image_data(
    df,
    images,
    distances,
):
    data = {
        "price": [],
        "images": [],
        "distances": [],
        "product_link": [],
    }
    for indx, image in enumerate(images):
        row = df[
            df.product_ids == image.split(".")[0]
        ]
        if not row.empty:
            data["price"].append(
                f"${row['product_prices'].values.tolist()[0]}"
            )
            val = requests.get(
                f"https:{row['image_links'].values.tolist()[0]}"
            ).content
            data["images"].append(
                Image.open(BytesIO(val))
            )
            data["distances"].append(
                distances[indx]
            )
            data["product_link"].append(
                row[
                    "product_page_links"
                ].values.tolist()[0]
            )
    return data


def gen_caption(data, idx):
    """Generate captions for the recommended products"""
    newline = "\n"
    caption = f"**Product Page:** https://www2.hm.com{data['product_link'][idx]} | **Price:** {data['price'][idx]}"
    return caption.split("|")


st.title(
    "**:blue[E2E: Fashion Recommendation System based on H&M data]**"
)
st.markdown("**_Code Available here:_**")


img = st.file_uploader(
    "Upload fashion-product", key="file_uploader"
)
if img is not None:
    frame = get_pandas_dataframe(
        path=os.getenv("DATAFRAME_PATH")
    )
    image = Image.open(img)
    op = get_results(image)
    data = get_image_data(
        frame,
        op["recommendations"],
        op["distances"][0],
    )
    idx = 0
    # Display images in a grid with their link and price
    if data:
        cols = st.columns(
            len(data["images"]), gap="small"
        )
        while idx < len(data["images"]):
            caption = gen_caption(data, idx)
            cols[idx].image(
                data["images"][idx],
                use_column_width=True,
            )
            for cap_idx in range(len(caption)):
                cols[idx].markdown(
                    f"{caption[cap_idx]}"
                )
            idx += 1
