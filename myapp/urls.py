from django.urls import path
from .views import home_view  # Đảm bảo import đúng tên hàm

urlpatterns = [
    path('', home_view, name='home'),  # Gọi đúng tên hàm
]
from django.urls import path
from .views import home_view, get_chart_data

urlpatterns = [
    path('', home_view, name='home'),
    path('chart-data/', get_chart_data, name='chart_data'),
]
