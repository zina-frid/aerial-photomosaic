import cgi
import threading
from http.server import BaseHTTPRequestHandler

from image_processing.utils import processing, clear_working_directory


ready_to_serve = False


class Server(BaseHTTPRequestHandler):

    with open('server/html_css/index.html', 'r') as file:
        index_page = file.read().rstrip()
        file.close()
    with open('server/html_css/stitching.html', 'r') as file:
        stitching_page = file.read().rstrip()
        file.close()
    with open('server/open.html', 'r') as file:
        open_str = file.read().rstrip()
        file.close()
    with open('server/html_css/styles.css', 'r') as file:
        css_page = file.read().rstrip()
        file.close()
    with open('server/src/favicon.ico', 'rb') as file:
        favicon = file.read()
        file.close()

    def handle_get(self):
        if self.path == "/":
            self.handle_get_index()
        elif self.path.endswith("/styles.css"):
            self.handle_get_stylesheet()
        elif self.path.endswith("/favicon.ico"):
            self.handle_get_favicon()
        elif self.path == "/stitch":
            self.handle_get_stitch()
        elif self.path == "/get_status":
            self.handle_get_status()
        elif self.path == "/result":
            self.handle_get_result()
        elif self.path == "/open":
            self.handle_get_open()
        elif self.path.endswith("add.png"):
            self.handle_get_add_image()
        elif self.path.endswith("folder.png"):
            self.handle_get_folder_image()
        elif "result_mosaic.png" in self.path:
            self.handle_get_result_image()
        elif self.path.endswith("map.html"):
            self.handle_get_map()
        elif self.path.endswith("script.js"):
            self.handle_get_script()

    # gets image on main page
    def handle_get_add_image(self):
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        with open('server/src/add.png', 'rb') as file:
            image = file.read()
            file.close()
        self.end_headers()
        self.wfile.write(image)

    # gets image on main page
    def handle_get_folder_image(self):
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        with open('server/src/folder.png', 'rb') as file:
            image = file.read()
            file.close()
        self.end_headers()
        self.wfile.write(image)

    # gets css
    def handle_get_stylesheet(self):
        self.send_response(200)
        self.send_header("Content-type", "text/css")
        self.end_headers()
        self.wfile.write(bytes(self.css_page, encoding='utf8'))

    # gets icon
    def handle_get_favicon(self):
        self.send_response(200)
        self.send_header("Content-type", "image/x-icon")
        self.end_headers()
        self.wfile.write(self.favicon)

    def handle_get_index(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(self.index_page, encoding='utf8'))

    def handle_get_stitch(self):
        clear_working_directory()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(self.stitching_page, encoding='utf8'))
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        if ctype == 'multipart/form-data':
            fields = cgi.parse_multipart(self.rfile, pdict)
            images = fields['upload-image']
            for index, img in enumerate(images):
                with open('server/uploaded_images/' + f'{index:04d}' + '.jpg', 'wb') as file:
                    file.write(img)
                    file.close()
        return

    def handle_get_open(self):
        clear_working_directory()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(self.open_str, encoding='utf8'))
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        if ctype == 'multipart/form-data':
            fields = cgi.parse_multipart(self.rfile, pdict)
            map_ = fields['folder-image'][0]
            with open('server/uploaded_images/' + 'map.html', 'wb') as file:
                file.write(map_)
                file.close()
        pass

    def do_GET(self):
        self.handle_get()

    # call main mosaic function
    def result(self):
        processing()
        global ready_to_serve
        ready_to_serve = True

    # post request
    def do_POST(self):
        if self.path == "/stitch":
            self.handle_get_stitch()
            thread = threading.Thread(target=self.result, args=[])
            thread.start()
        elif self.path == "/open":
            self.handle_get_open()
            self.handle_get_result()
        else:
            print("no route")

    # gets html result
    def handle_get_result(self):
        global ready_to_serve
        if ready_to_serve:
            with open('server/result.html', 'r') as file:
                res = file.read()
                file.close()
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(res, encoding='utf8'))
            ready_to_serve = False
        else:
            print("error")

    # gets result image
    def handle_get_result_image(self):
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        with open('server/uploaded_images/result_mosaic.png', 'rb') as file:
            image = file.read()
            file.close()
        self.end_headers()
        self.wfile.write(image)

    def handle_get_map(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        with open('server/uploaded_images/map.html', 'r') as file:
            res = file.read()
            file.close()
        self.end_headers()
        self.wfile.write(bytes(res, encoding='utf8'))

    # gets script for zoom
    def handle_get_script(self):
        self.send_response(200)
        self.send_header("Content-type", "application/javascript")
        with open('server/script.js', 'r') as file:
            script = file.read()
            file.close()
        self.end_headers()
        self.wfile.write(bytes(script, encoding='utf8'))

    def handle_get_status(self):
        if ready_to_serve:
            self.send_response(200)
        else:
            self.send_response(201)

        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("", encoding='utf8'))
