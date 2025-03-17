document.getElementById('generateCaptionBtn').onclick = async () => {
  const imageInput = document.getElementById('imageInput').files[0];
  if (!imageInput) {
      alert("Upload an image first!");
      return;
  }

  const formData = new FormData();
  formData.append('image', imageInput);

  const response = await fetch('/upload', { method: 'POST', body: formData });
  const data = await response.json();

  document.getElementById('captionOutput').innerText = data.caption;
};

document.getElementById('imageInput').onchange = evt => {
  const [file] = event.target.files;
  const reader = new FileReader();
  reader.onload = e => {
      document.getElementById('imagePreview').src = e.target.result;
      document.getElementById('imagePreview').style.display = 'block';
  };
  reader.readAsDataURL(file);
};
