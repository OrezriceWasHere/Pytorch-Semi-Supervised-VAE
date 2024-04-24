from clearml import Task

import params

ALLOW_CLEARML = True
RUNNING_REMOTE = True

if ALLOW_CLEARML:
    execution_task = Task.init(project_name="Pytorch Semi Supervised VAE",
                               task_name="First version",
                               task_type=Task.TaskTypes.optimizer,
                               reuse_last_task_id=False)
    execution_task.set_parameters_as_dict(params.Params.__dict__)

    if RUNNING_REMOTE:
        execution_task.execute_remotely(queue_name="gpu", exit_process=True)



def clearml_display_image(image, description):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_image(description, iteration=execution_task.get_last_iteration(), image=image)

def add_point_to_graph(title, series, x, y):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_scalar(title=title, series=series, value=y, iteration=x)

