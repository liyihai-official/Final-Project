from tensorflow import reduce_mean, abs, maximum, reduce_max
from numpy import sqrt
# Define custom loss function
def custom_loss(model, x, y, training):
    y_ = model(x, training=training)
    
    return reduce_mean((y_- y)**2)

def approx_err(model, x, u_reference, mc_rounds):
    l1_err, l2_err, li_err = 0., 0., 0.
    rel_l1_err, rel_l2_err, rel_li_err = 0., 0., 0.
    
    for _ in range(mc_rounds):
        u_approx = model(x, training=False)
        err = abs(u_approx - u_reference)
        
        l1_err += reduce_mean(err)
        l2_err += reduce_mean(err**2)
        li_err = maximum(li_err, reduce_mean(err))
        
        rel_err = err / maximum(u_reference, 1e-8)
        rel_l1_err += reduce_mean(rel_err)
        rel_l2_err += reduce_mean(rel_err**2)
        rel_li_err = maximum(rel_li_err, reduce_max(rel_err))
    
    l1_err /= mc_rounds
    l2_err = sqrt(l2_err / mc_rounds)
    rel_l1_err /= mc_rounds
    rel_l2_err = sqrt(rel_l2_err / mc_rounds)
    
    return err, l1_err, l2_err, li_err, rel_l1_err, rel_l2_err, rel_li_err
