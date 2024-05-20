(ns scicloj.ml.deep-diamond.classification
  (:require
   [tablecloth.api :as tc]
   [scicloj.ml.deep-diamond.boston-housing-data :as boston-data]
   [tablecloth.column.api :as tcc]
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.deep-diamond.infer :as infer]

   [uncomplicate.diamond.dnn :as dnn]
   [uncomplicate.diamond.internal.cost                      :refer [binary-accuracy!] :as dd-cost]
   [uncomplicate.diamond.internal.neanderthal.factory       :refer [neanderthal-factory]]
   [uncomplicate.diamond.internal.protocols                 :as dd-proto]
   [uncomplicate.diamond.tensor  :as dd-tz]
   [uncomplicate.neanderthal.core  :as nea-core]
   [uncomplicate.neanderthal.native :as nea-native]
   [uncomplicate.commons.core  :as uc]))

(set! *print-length* 128)

(defn pr-edn-str [& xs]
  (binding [*print-length* nil
            *print-dup* nil
            *print-level* nil
            *print-readably* true]
    (apply pr-str xs)))

(defn train [feature-ds target-ds]
  (def feature-ds feature-ds)
  (uc/let-release [
                   fact (neanderthal-factory)

                   x-tz-train (dd-tz/tensor fact [boston-data/train-size boston-data/max-vocab] :float :nc)
                   x-mb-tz-train (dd-tz/tensor fact [boston-data/mb-size boston-data/max-vocab] :float :nc)
                   y-tz-train (dd-tz/tensor fact [boston-data/train-size 1] :float :nc)
                   y-mb-tz-train (dd-tz/tensor fact [boston-data/mb-size 1] :float :nc)
                   x-batcher-train (dd-tz/batcher x-tz-train x-mb-tz-train)
                   y-batcher-train (dd-tz/batcher y-tz-train y-mb-tz-train)


                   net-bp (dnn/network fact
                                       x-mb-tz-train
                                       [(dnn/fully-connected [16] :relu)
                                        (dnn/fully-connected [16] :relu)
                                        (dnn/fully-connected [1] :sigmoid)])
                   net (dnn/init! (net-bp x-mb-tz-train :adam))

                   crossentropy-cost (dnn/cost net y-mb-tz-train :crossentropy)

                   _ (nea-core/transfer! (apply concat (-> feature-ds tc/rows)) (nea-core/view-vctr x-tz-train))
                   _ (nea-core/transfer! (apply concat (-> target-ds tc/rows)) (nea-core/view-vctr y-tz-train))
                   _ (dnn/train! net x-batcher-train y-batcher-train crossentropy-cost 5 [])

                   prediction (dnn/infer! net x-tz-train)

                   binary-accuracy (binary-accuracy! y-tz-train prediction)]

                  (println :binary-accuracy-train binary-accuracy)

                  {:params (for [layer net
                                 params (dd-proto/parameters layer)]
                             (seq params))
                   :binary-accuracy-train binary-accuracy}))
  

(defn- iter [data]
  (let [it (.iterator (sequence data))]
    #(when (.hasNext it) (.next it))))

(defn predict [feature-ds all-params]
  (uc/let-release [

                   fact (neanderthal-factory)

                   x-tz-test (dd-tz/tensor fact [boston-data/test-size boston-data/max-vocab] :float :nc)
                   x-mb-tz-test (dd-tz/tensor fact [boston-data/mb-size boston-data/max-vocab] :float :nc)


                   x-batcher-test (dd-tz/batcher x-tz-test x-mb-tz-test)


                   net-bp (dnn/network fact
                                       x-mb-tz-test
                                       [(dnn/fully-connected [16] :relu)
                                        (dnn/fully-connected [16] :relu)
                                        (dnn/fully-connected [1] :sigmoid)])


                   params-iter (iter all-params)
                   net (dnn/init! (net-bp x-mb-tz-test :adam)
                                  (fn [t]
                                    (let [shape (dd-tz/shape t)]
                                      (println shape)
                                      (nea-core/transfer!
                                       ;; (repeatedly (* (first shape) (get 0 shape 1)) rand)
                                       (params-iter)
                                       t)
                                      t)))

                   _ (nea-core/transfer!
                      (apply concat (-> feature-ds tc/rows))
                      (nea-core/view-vctr x-tz-test))

                   prediction (dnn/infer! net x-tz-test)]


    (println :prediction-test prediction)

    (tc/dataset
     {:prediction (seq prediction)})))
      



(ml/define-model! :deep-diamond/classification
  (fn [feature-ds target-ds options]
    (train feature-ds target-ds))


  (fn [feature-ds thawed-model {:keys [options model-data target-categorical-maps] :as model}]
    (predict
     feature-ds
     (:params model-data)))
  {})
