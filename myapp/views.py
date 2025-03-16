from django.shortcuts import render
from .forms import CalculationForm
from .models import Calculation
from django.core.files.storage import FileSystemStorage  # Thêm để lưu file ảnh

def home_view(request):
    
    result = None
    image_url = None  # Biến lưu đường dẫn ảnh

    form = CalculationForm(request.POST or None)


    if request.method == "POST" and form.is_valid():
        ts1 = form.cleaned_data['ts1']
        ts2 = form.cleaned_data['ts2']
        ts3 = form.cleaned_data['ts3']
        ts4 = form.cleaned_data['ts4']
        operation = form.cleaned_data['operation']

        if operation == "Cộng":
            result = ts1 + ts2 + ts3 + ts4
        elif operation == "Trừ":
            result = ts1 - ts2 - ts3 - ts4
        elif operation == "Nhân":
            result = ts1 * ts2 * ts3 * ts4
        else:
            result = None  # Tránh chia cho 0
        

        # Lưu kết quả vào database
        Calculation.objects.create(
            ts1=ts1, ts2=ts2, ts3=ts3, ts4=ts4, operation=operation, result=result
        )

         # Xử lý ảnh tải lên
        if 'image' in request.FILES:
            image = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            image_url = fs.url(filename)  # Lấy đường dẫn ảnh
            
            
    return render(request, 'myapp/home.html', {'form': form, 'result': result,  'image_url': image_url})
  
from django.http import JsonResponse
import random

def get_chart_data(request):
    epochs = list(range(1, 20))  # 50 epochs

    # Tạo dữ liệu ngẫu nhiên để mô phỏng
    train_loss = [random.uniform(0.2, 0.6) for _ in epochs]
    test_loss = [random.uniform(0.2, 0.6) for _ in epochs]

    train_accuracy = [random.uniform(0.75, 0.95) for _ in epochs]
    test_accuracy = [random.uniform(0.75, 0.95) for _ in epochs]

    data = {
        "epochs": epochs,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }
    return JsonResponse(data)

def my_view(request):
    if request.method == "POST":
        # Xử lý dữ liệu
        pass
    return render(request, "home.html")
