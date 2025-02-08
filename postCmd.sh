curl -X 'POST' \
  'http://127.0.0.1:8000/process_image/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/workspaces/LicensePlateJP/image/image7.png' \
  -F 'measure=true'
