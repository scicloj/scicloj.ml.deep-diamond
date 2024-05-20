(ns scicloj.ml.deep-diamond.classification-test
  (:require


   [clojure.test :refer [deftest is]]
   [scicloj.ml.deep-diamond.classification]
   [scicloj.ml.deep-diamond.boston-housing-data :as boston-data]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [tech.v3.dataset.modelling :as ds-mod]
   [tablecloth.api :as tc]
   [tablecloth.column.api :as tcc]))


(def boston-data
  (boston-data/boston-data 10000 50000 25000))




(def train-ds
  (-> boston-data
      :train-ds
      (ds-mod/set-inference-target :y)))




(def test-ds
  (-> boston-data
      :test-ds
      (tc/drop-columns [:y])))


(deftest boston-accurcay


  (time
   (let [model (ml/train train-ds {:model-type :deep-diamond/classification})
         prediction (ml/predict test-ds model)]


     (is (< 0.83
            (loss/classification-accuracy
             (-> boston-data :test-ds :y)
             (tcc/round (:prediction prediction))))))))
