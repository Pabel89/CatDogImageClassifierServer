# Informationen
Dockerisierter Python Webserver um Bilder von Hunden oder Katzen zum Bildklassifierungs CNN zu schicken. Per Post Request und **content-type: multipart/form-data**
kann eine Bilddatei im Body mit Key: file zum Server an den Endpoint: **/uploads** geschickt werden. Der Server liefert eine Response, ob auf dem Bild ein Hund oder
eine Katze zu sehen ist in Form eines JSONs. Das JSON Attribut **message** enthaelt den Wert.

# Umgebung
Eine Dockerengine wird benötigt zum Beispiel Docker Desktop für Windows.

# Ausfuehren 
Befehle: 
**docker-compose build**  - Bauen des Containers
**docker-compose up** - Start des Containers
**docker-compose down** - Stopp des Containers und loeschen des Containers