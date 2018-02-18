#!/usr/bin/env python3

import web
import tensorflow as tf
from prognosis import probability_of_survival, graph

urls = (
    '/prognosis', 'prognosis'
)

class prognosis:
    def GET(self):
        url_input = web.input()
        patient = (
            int(url_input.age),
            int(url_input.gender),
            int(url_input.face),
            int(url_input.arm),
            int(url_input.leg),
            int(url_input.dysphasia),
            int(url_input.hemianopia),
            int(url_input.visuospatial),
            int(url_input.cerebellar),
            int(url_input.aspirin),
            int(url_input.carotid),
            int(url_input.thromb),
            int(url_input.stroke_14),
            int(url_input.haem_14),
            int(url_input.pulm_14)
            )
        print(patient)

        with graph.as_default():
            return str(probability_of_survival(*patient))

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run(port=8888)
