document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
  
    const formData = new FormData(event.target);
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData,
    });
  
    const result = await response.json();
    const resultDiv = document.getElementById('result');
    if (response.ok) {
      resultDiv.innerHTML = `<p>${result.message}</p><pre>${JSON.stringify(result.result, null, 2)}</pre>`;
    } else {
      resultDiv.innerHTML = `<p>Error: ${result.message}</p>`;
    }
  });