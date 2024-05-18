(ns scicloj.ml.deep-diamond.classification-test
  (:require

   [scicloj.ml.deep-diamond.classification]
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset.modelling :as ds-mod]
   [tablecloth.api :as tc]))


(def ds
  (->
   (tc/dataset {:a [1]
                :b [1]})
   (ds-mod/set-inference-target :b)))




(run!
 (fn [_]
   (let [model (ml/train ds {:model-type :deep-diamond/classification})
         prediction (ml/predict ds model)]
     (println prediction)))

          
 (range 50))
