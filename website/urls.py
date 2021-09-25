from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('user_classifier.urls')),
    path('admin_access/', include('admin_classifier.urls'))
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, documnet_root=settings.MEDIA_ROOT)
