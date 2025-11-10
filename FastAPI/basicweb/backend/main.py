from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Import CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- Cấu hình CORS ---
# Đây là bước BẮT BUỘC để frontend (CSR) có thể gọi được API
origins = [
    "*",  # Cho phép tất cả các nguồn (origins) - không an toàn cho production
    # có thể chỉ định rõ: "http://127.0.0.1:5500" (nếu dùng Live Server)
    # hoặc "null" (nếu mở file .html trực tiếp)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các header
)


# --- Pydantic Models (Schemas) ---
# Mô hình dữ liệu cho item
class Item(BaseModel):
    name: str
    description: Optional[str] = None

# --- "Cơ sở dữ liệu" giả lập (in-memory) ---
# Một danh sách Python đơn giản để lưu trữ dữ liệu
db: List[Item] = [
    Item(name="Sản phẩm 1", description="Mô tả cho sản phẩm đầu tiên"),
    Item(name="Sản phẩm 2", description="Mô tả cho sản phẩm thứ hai"),
]

# --- Các API Endpoints ---

@app.get("/api/items", response_model=List[Item])
async def get_all_items():
    """
    Endpoint GET: Trả về toàn bộ danh sách items.
    """
    return db

@app.post("/api/items", response_model=Item)
async def create_new_item(item: Item):
    """
    Endpoint POST: Nhận dữ liệu JSON, thêm vào db và trả về item vừa tạo.
    """
    db.append(item)
    return item

@app.get("/")
async def root():
    return {"message": "Đây là backend FastAPI. Hãy mở file index.html để xem frontend."}