import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class ExtrapMethod(Enum):
    
    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class Interpolator1D(ABC):

    def __init__(self,
                 axis1 : np.ndarray, 
                 values : np.ndarray, 
                 interpolation_method : InterpMethod,
                 extrpolation_method : ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrpolation_method
        self.length_ = len(self.axis1)

    @abstractmethod
    def interpolate(self, x : float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x : float, end_x : float):
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x : float):
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        pass
    
    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_
    
    @property
    def values(self) -> np.ndarray:
        return self.values_
    
    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()
    
    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()

class Interpolator1DPCP(Interpolator1D):

    def __init__(self, axis1: np.ndarray, values: np.ndarray, extrpolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.LINEAR, extrpolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        if x < self.axis1_[0]:
            return self.values_[0]
        if x >= self.axis1_[-1]:
            return self.values_[-1]
        k = int(np.searchsorted(self.axis1_,x,side="right"))
        return self.values_[k]
    
    def gradient_wrt_ordinate(self, x : float):
        grad = np.zeros(self.length_, dtype=float)
        if x < self.axis1_[0]:
            grad[0] = 1.0
            return grad
        if x >= self.axis1_[-1]:
            grad[-1] = 1.0
            return grad
        k = int(np.searchsorted(self.axis1_,x,side="right"))
        grad[k] = 1.0
        return grad

    def integrate(self, start_x : float, end_x : float):
        if self.length_==1:
            return self.values_[0] * (end_x - start_x)
        
        if start_x == end_x:
            return 0.0
        
        # direction
        sign = 1.0
        a = float(start_x)
        b = float(end_x)
        if a > b:
            a,b = b,a
            sign = -1.0

        # calculate from left to right
        total = 0.0
        if a < self.axis1_[0]:
            total += self.values_[0] * (min(b,self.axis1_[0]) - a)
            a = min(b,self.axis1_[0])
        
        if a < b and a < self.axis1_[-1]:
            i = int(np.searchsorted(self.axis1_, a, side="right"))
            if i < 1:
                i=1
            if i > (self.length_-1):
                i = self.length_ -1
            while a < b and i <= (self.length_-1):
                seg_end = self.axis1_[i]
                total += self.values_[i] * (min(b,seg_end) - a)
                a = min(b,seg_end)
                if a>=b or a>=self.axis1_[-1]:
                    break
                i+=1
        
        if a < b and a >= self.axis1_[-1]:
            total += self.values_[-1] * (b-a)
        
        return total * sign

    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        grad = np.zeros(self.length_,dtype=float)
        if self.length_==1:
            grad[0] = (end_x - start_x)
            return grad
        
        if start_x == end_x:
            return grad
        
        # direction
        sign = 1.0
        a = float(start_x)
        b = float(end_x)
        if a > b:
            a,b = b,a
            sign = -1.0

        # calculate from left to right
        if a < self.axis1_[0]:
            grad[0] += (min(b, self.axis1_[0]) - a)
            a = min(b, self.axis1_[0])
        
        if a < b and a < self.axis1_[-1]:
            i = int(np.searchsorted(self.axis1_, a, side="right"))
            if i < 1:
                i = 1
            if i > self.length_ - 1:
                i = self.length_ - 1
            while a < b and i <= self.length_ - 1:
                seg_end = self.axis1_[i]
                grad[i] += (min(b, seg_end) - a)
                a = min(b, seg_end)
                if a>=b or a>=self.axis1_[-1]:
                    break
                i += 1
        
        if a < b and a >= self.axis1_[-1]:
            grad[-1] += (b - a)

        return sign * grad

class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1 : np.ndarray | List, 
                               values : np.ndarray | List, 
                               interpolation_method : InterpMethod,
                               extrpolation_method : ExtrapMethod):


        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)
    
        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrpolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')
