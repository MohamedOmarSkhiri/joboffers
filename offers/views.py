import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.http import JsonResponse
from offers.utils import predict_from_pdf

def upload_file(request):
    uploaded_file_url = None
    prediction = None

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_url = fs.url(filename)

        file_path = os.path.join(settings.MEDIA_ROOT, filename)

        try:
            prediction = predict_from_pdf(file_path)
        except Exception as e:
            prediction = f"Error during prediction: {str(e)}"

    return render(request, 'upload.html', {
        'file_url': uploaded_file_url,
        'prediction': prediction
    })
