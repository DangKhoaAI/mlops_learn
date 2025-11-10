// Chờ cho toàn bộ cây DOM được tải xong trước khi chạy code
document.addEventListener("DOMContentLoaded", () => {

    // Địa chỉ API backend
    const API_BASE_URL = "http://127.0.0.1:8000";

    const itemListEl = document.getElementById("item-list");
    const itemFormEl = document.getElementById("add-item-form");

    // --- Hàm 1: Tải và Render danh sách items (GET) ---
    async function fetchAndRenderItems() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/items`);
            if (!response.ok) throw new Error("Network response was not ok");
            
            const items = await response.json();

            // Xóa nội dung "Đang tải..."
            itemListEl.innerHTML = ""; 

            if (items.length === 0) {
                itemListEl.innerHTML = "<li>Chưa có sản phẩm nào.</li>";
                return;
            }

            // Render từng item lấy từ API
            items.forEach(item => {
                const li = document.createElement("li");
                li.textContent = `${item.name} - (Mô tả: ${item.description || 'N/A'})`;
                itemListEl.appendChild(li);
            });

        } catch (error) {
            console.error("Lỗi khi tải items:", error);
            itemListEl.innerHTML = "<li>Lỗi! Không thể tải dữ liệu từ server.</li>";
        }
    }

    // --- Hàm 2: Xử lý thêm item mới (POST) ---
    itemFormEl.addEventListener("submit", async (event) => {
        event.preventDefault(); // Ngăn form submit và tải lại trang

        const nameInput = document.getElementById("item-name");
        const descInput = document.getElementById("item-desc");

        const newItem = {
            name: nameInput.value,
            description: descInput.value
        };

        try {
            const response = await fetch(`${API_BASE_URL}/api/items`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(newItem),
            });

            if (response.ok) {
                // Thêm thành công!
                // Xóa form
                nameInput.value = "";
                descInput.value = "";
                // Tải lại danh sách để hiển thị item mới
                fetchAndRenderItems(); 
            } else {
                alert("Thêm sản phẩm thất bại.");
            }

        } catch (error) {
            console.error("Lỗi khi thêm item:", error);
            alert("Lỗi kết nối khi thêm item.");
        }
    });


    // --- Chạy lần đầu tiên ---
    // Vì code này đã nằm trong 'DOMContentLoaded', 
    // chúng ta có thể gọi hàm fetch ngay lập tức
    fetchAndRenderItems();

});