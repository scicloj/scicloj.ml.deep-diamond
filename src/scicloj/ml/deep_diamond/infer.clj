(ns scicloj.ml.deep-diamond.infer
  (:require
   [scicloj.ml.deep-diamond.text-tools :as text]
   [tablecloth.api :as tc]
   [uncomplicate.commons.core :refer [let-release]]
   [uncomplicate.diamond.dnn :refer [fully-connected infer! init! network]]
   [uncomplicate.diamond.internal.cost :refer [binary-accuracy!]]
   [uncomplicate.diamond.internal.neanderthal.factory
    :refer [neanderthal-factory]]
   [uncomplicate.diamond.tensor :refer [batcher shape tensor]]
   [uncomplicate.neanderthal.core :refer [ncols transfer! view-vctr native]]))


(defn iter [data]
  (let [it (.iterator (sequence data))]
    #(when (.hasNext it) (.next it))))


;; (def all-params
;;   (clojure.edn/read-string (slurp "params.edn")))



(defn predict [all-params]
  (let-release [

                x-test (first text/data-test)
                y-test (second text/data-test)


                test-size (-> x-test ncols)

                fact (neanderthal-factory)



                x-tz-test (tensor fact [test-size text/max-vocab] :float :nc)
                x-mb-tz-test (tensor fact [text/mb-size text/max-vocab] :float :nc)
                y-tz-test (tensor fact [test-size 1] :float :nc)
                y-mb-tz-test (tensor fact [text/mb-size 1] :float :nc)
                x-batcher-test (batcher x-tz-test x-mb-tz-test)
                y-batcher-test (batcher y-tz-test y-mb-tz-test)

                net-bp (network fact
                                x-mb-tz-test
                                [(fully-connected [16] :relu)
                                 (fully-connected [16] :relu)
                                 (fully-connected [1] :sigmoid)])




                params-iter (iter all-params)
                net (init! (net-bp x-mb-tz-test :adam)
                           (fn [t]
                             (let [shape (shape t)]
                               (println shape)
                               (transfer!
                                ;; (repeatedly (* (first shape) (get 0 shape 1)) rand)
                                (params-iter)
                                t)
                               t)))

                _ (transfer! x-test (view-vctr x-tz-test))
                _ (transfer! y-test (view-vctr y-tz-test))

                prediction (native (infer! net (x-batcher-test)))]





    (println {:binary-accuracy-test
              (binary-accuracy!
               prediction
               (y-batcher-test))})

    (tc/dataset {:prediction (seq prediction)})))
