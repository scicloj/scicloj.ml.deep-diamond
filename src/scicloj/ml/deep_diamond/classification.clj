(ns scicloj.ml.deep-diamond.classification
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.deep-diamond.infer :as infer]
   [scicloj.ml.deep-diamond.text-tools :as text]
   [uncomplicate.diamond.dnn
    :refer [cost fully-connected infer! init! network train!]]
   [uncomplicate.diamond.internal.cost :refer [binary-accuracy!]]
   [uncomplicate.diamond.internal.neanderthal.factory
    :refer [neanderthal-factory]]
   [uncomplicate.diamond.internal.protocols :refer [parameters]]
   [uncomplicate.diamond.tensor :refer [batcher tensor]]
   [uncomplicate.neanderthal.core :refer [transfer! view-vctr]]
   [uncomplicate.commons.core :refer [let-release]]))


(defn pr-edn-str [& xs]
    (binding [*print-length* nil
              *print-dup* nil
              *print-level* nil
              *print-readably* true]
      (apply pr-str xs)))

(defn train []
  (let-release [

                fact (neanderthal-factory)
                x-train (first text/data-train)
                y-train (second text/data-train)



                x-tz-train (tensor fact [text/train-size text/max-vocab] :float :nc)
                x-mb-tz-train (tensor fact [text/mb-size text/max-vocab] :float :nc)
                y-tz-train (tensor fact [text/train-size 1] :float :nc)
                y-mb-tz-train (tensor fact [text/mb-size 1] :float :nc)
                x-batcher-train (batcher x-tz-train x-mb-tz-train)
                y-batcher-train (batcher y-tz-train y-mb-tz-train)


                net-bp (network fact
                                x-mb-tz-train
                                [(fully-connected [16] :relu)
                                 (fully-connected [16] :relu)
                                 (fully-connected [1] :sigmoid)])
                net (init! (net-bp x-mb-tz-train :adam))

                crossentropy-cost (cost net y-mb-tz-train :crossentropy)




                _ (transfer! x-train (view-vctr x-tz-train))
                _ (transfer! y-train (view-vctr y-tz-train))
                _ (train! net x-batcher-train y-batcher-train crossentropy-cost 5 [])

                prediction (infer! net x-mb-tz-train)
                binary-accuracy (binary-accuracy! y-mb-tz-train prediction)]


    (println :binary-accuracy binary-accuracy)

    {:params (for [layer net
                   params (parameters layer)]
               (seq params))
     :binary-accuracy-train binary-accuracy}))
  







(ml/define-model! :deep-diamond/classification
  (fn [feature-ds target-ds options]
    (train))


  (fn [feature-ds thawed-model {:keys [options model-data target-categorical-maps] :as model}]
    (def model-data model-data)
    (infer/predict (:params model-data)))
  {})
