(ns scicloj.ml.deep-diamond.infer
  (:require

   [tablecloth.api :as tc]
   [uncomplicate.commons.core :refer [let-release]]
   [uncomplicate.diamond.dnn :refer [fully-connected infer! init! network]]
   [uncomplicate.diamond.internal.cost :refer [binary-accuracy!]]
   [uncomplicate.diamond.internal.neanderthal.factory
    :refer [neanderthal-factory]]
   [uncomplicate.diamond.tensor :refer [batcher shape tensor]]
   [uncomplicate.neanderthal.core :refer [ncols transfer! view-vctr native]]))





;; (def all-params
;;   (clojure.edn/read-string (slurp "params.edn")))
