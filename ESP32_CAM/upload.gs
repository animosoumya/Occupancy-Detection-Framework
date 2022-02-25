// Google Script Code

function doPost(e) {
  var data = Utilities.base64Decode(e.parameters.data);
  var nombreArchivo = Utilities.formatDate(new Date(), "GMT-3", "yyyyMMdd_HHmmss")+".jpg";
  var blob = Utilities.newBlob(data, e.parameters.mimetype, nombreArchivo );
  
  
   // Save the photo to Google Drive
  var folder, folders = DriveApp.getFoldersByName("ESP32-CAM");
  if (folders.hasNext()) {
    folder = folders.next();
  } else {
    folder = DriveApp.createFolder("ESP32-CAM"); // if folder doesn't exist, create a new one
  }
  var file = folder.createFile(blob); 
  return ContentService.createTextOutput('Completo')
}
