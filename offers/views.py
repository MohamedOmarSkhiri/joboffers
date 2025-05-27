import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from offers.utils import predict_from_pdf

def upload_file(request):
    uploaded_file_url = None
    formatted_prediction = None  # changed variable name for clarity

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_url = fs.url(filename)

        try:
            prediction_result = predict_from_pdf(filename)  # e.g., {'prediction': 'Real', 'probability_fake': 0.43}

            prediction = prediction_result.get('prediction', 'Unknown')
            probability_fake = prediction_result.get('probability_fake', 0)

            try:
                probability_float = float(probability_fake)
            except:
                probability_float = 0

            probability_percentage = probability_float * 100

            formatted_prediction = f"prediction: {prediction}, probability is fake: {probability_percentage:.0f}%"

        except Exception as e:
            formatted_prediction = f"Error: {str(e)}"

    return render(request, 'upload.html', {
        'file_url': uploaded_file_url,
        'prediction': formatted_prediction
    })
