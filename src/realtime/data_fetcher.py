"""
实时数据获取器 - 自动获取最新的彩票开奖数据
"""

import requests
import json
import re
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
from queue import Queue

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    实时数据获取器
    支持多数据源，自动获取最新开奖数据
    """
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        初始化数据获取器
        
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # 数据源配置 - 使用500彩票网和其他可靠源
        self.data_sources = {
            'shuangseqiu': {
                'primary': {
                    'name': '500彩票网双色球历史数据',
                    'base_url': 'https://datachart.500.com/ssq/history/',
                    'history_url': 'https://datachart.500.com/ssq/history/history.shtml',
                    'data_url': 'https://datachart.500.com/ssq/history/newinc/history.php',
                    'method': 'GET',
                    'parser': 'html',
                    'encoding': 'gb2312'
                },
                'secondary': {
                    'name': '新浪彩票接口',
                    'url': 'https://interface.sina.cn/pc_api/lottery_interface.php',
                    'method': 'GET',
                    'parser': 'json',
                    'params': {
                        'lottery_type': 'ssq',
                        'num': '30'
                    }
                },
                'tertiary': {
                    'name': 'OpenLottery免费接口',
                    'url': 'https://api.openlottery.cn/api/welfare/ssq',
                    'method': 'GET',
                    'parser': 'json'
                },
                'fallback': {
                    'name': '本地缓存',
                    'url': 'cache://shuangseqiu',
                    'method': 'CACHE'
                }
            },
            'daletou': {
                'primary': {
                    'name': '500彩票网大乐透历史数据',
                    'base_url': 'https://datachart.500.com/dlt/history/',
                    'history_url': 'https://datachart.500.com/dlt/history/history.shtml',
                    'data_url': 'https://datachart.500.com/dlt/history/newinc/history.php',
                    'method': 'GET',
                    'parser': 'html',
                    'encoding': 'gb2312'
                },
                'secondary': {
                    'name': '新浪彩票接口',
                    'url': 'https://interface.sina.cn/pc_api/lottery_interface.php',
                    'method': 'GET',
                    'parser': 'json',
                    'params': {
                        'lottery_type': 'dlt',
                        'num': '30'
                    }
                },
                'tertiary': {
                    'name': 'OpenLottery免费接口',
                    'url': 'https://api.openlottery.cn/api/welfare/dlt',
                    'method': 'GET',
                    'parser': 'json'
                },
                'fallback': {
                    'name': '本地缓存',
                    'url': 'cache://daletou',
                    'method': 'CACHE'
                }
            }
        }
        
        # 缓存最新数据
        self.latest_data_cache = {}
        self.cache_timestamp = {}
        
        # 线程锁
        self.lock = threading.Lock()
    
    def get_latest_data(self, lottery_type: str, periods: int = 1) -> List[Dict]:
        """
        获取最新开奖数据
        
        Args:
            lottery_type: 彩票类型 ('双色球' 或 '大乐透')
            periods: 获取期数
            
        Returns:
            开奖数据列表
        """
        try:
            logger.info(f"开始获取{lottery_type}最新{periods}期数据...")
            
            lottery_code = 'shuangseqiu' if lottery_type == '双色球' else 'daletou'
            
            # 尝试多个数据源
            for source_type in ['primary', 'secondary', 'tertiary', 'fallback']:
                try:
                    data = self._fetch_from_source(lottery_code, source_type, periods)
                    if data:
                        logger.info(f"成功从{source_type}源获取{len(data)}期数据")
                        self._update_cache(lottery_type, data)
                        return data
                except Exception as e:
                    logger.warning(f"从{source_type}源获取数据失败: {e}")
                    continue
            
            # 所有源都失败，返回缓存数据
            return self._get_cached_data(lottery_type, periods)
            
        except Exception as e:
            logger.error(f"获取最新数据失败: {e}")
            return []
    
    def get_historical_data(self, lottery_type: str, start_period: str, 
                          end_period: str) -> List[Dict]:
        """
        获取历史数据范围
        
        Args:
            lottery_type: 彩票类型
            start_period: 开始期号
            end_period: 结束期号
            
        Returns:
            历史数据列表
        """
        try:
            logger.info(f"获取{lottery_type}历史数据: {start_period} - {end_period}")
            
            lottery_code = 'shuangseqiu' if lottery_type == '双色球' else 'daletou'
            
            # 使用主要数据源获取历史数据
            data = self._fetch_historical_range(lottery_code, start_period, end_period)
            
            if data:
                logger.info(f"成功获取{len(data)}期历史数据")
                return data
            else:
                logger.warning("未获取到历史数据")
                return []
                
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return []
    
    def check_new_results(self, lottery_type: str, last_known_period: str) -> Optional[Dict]:
        """
        检查是否有新的开奖结果
        
        Args:
            lottery_type: 彩票类型
            last_known_period: 已知的最新期号
            
        Returns:
            新的开奖数据，如果没有则返回None
        """
        try:
            latest_data = self.get_latest_data(lottery_type, 1)
            
            if latest_data and len(latest_data) > 0:
                latest_period = latest_data[0].get('period', '')
                
                if latest_period and latest_period != last_known_period:
                    logger.info(f"发现新开奖结果: {latest_period}")
                    return latest_data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"检查新结果失败: {e}")
            return None
    
    def _fetch_from_source(self, lottery_code: str, source_type: str, 
                          periods: int) -> List[Dict]:
        """从指定数据源获取数据"""
        source_config = self.data_sources[lottery_code][source_type]
        method = source_config.get('method', 'GET')
        parser = source_config.get('parser', 'json')
        
        if method == 'CACHE':
            return self._fetch_from_local_cache(lottery_code, periods)
        elif method in ['GET', 'POST']:
            if parser == 'json':
                return self._fetch_from_api(lottery_code, source_config, periods)
            elif parser == 'html':
                if '500.com' in source_config.get('base_url', ''):
                    return self._fetch_from_500wan_correct(lottery_code, source_config, periods)
                else:
                    return self._fetch_from_500wan_html(lottery_code, source_config, periods)
            elif parser == 'xml':
                return self._fetch_from_500wan_xml(lottery_code, source_config, periods)
        
        return []
    
    def _fetch_from_500wan_correct(self, lottery_code: str, source_config: Dict, periods: int) -> List[Dict]:
        """使用正确的方法从500彩票网获取数据（基于示例代码）"""
        try:
            # 获取彩票类型的代码
            lottery_name = 'ssq' if lottery_code == 'shuangseqiu' else 'dlt'
            
            # 先获取最新期号
            current_number = self._get_current_number_500wan(lottery_name, source_config)
            if not current_number:
                logger.warning(f"无法获取{lottery_name}最新期号")
                return []
            
            logger.info(f"500彩票网{lottery_name}最新期号: {current_number}")
            
            # 计算起始期号
            start_number = max(1, int(current_number) - periods + 1)
            
            # 获取历史数据
            return self._spider_500wan_data(lottery_name, start_number, current_number, source_config)
            
        except Exception as e:
            logger.error(f"从500彩票网获取数据失败: {e}")
            return []
            
    def _get_current_number_500wan(self, lottery_name: str, source_config: Dict) -> str:
        """获取500彩票网最新期号"""
        try:
            history_url = source_config['history_url']
            
            # 使用gb2312编码
            response = self._make_request(history_url)
            if not response:
                return None
            
            # 设置正确的编码
            response.encoding = source_config.get('encoding', 'gb2312')
            
            if BS4_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找期号输入框
                current_num_input = soup.find('input', {'id': 'end'})
                if current_num_input and current_num_input.get('value'):
                    return current_num_input['value']
                
                # 如果没有找到，尝试其他方法
                wrap_datachart = soup.find('div', class_='wrap_datachart')
                if wrap_datachart:
                    end_input = wrap_datachart.find('input', id='end')
                    if end_input and end_input.get('value'):
                        return end_input['value']
            
            # 如果BeautifulSoup失败，使用正则表达式
            import re
            pattern = r'<input[^>]*id=["\']end["\'][^>]*value=["\'](\d+)["\']'
            match = re.search(pattern, response.text)
            if match:
                return match.group(1)
            
            return None
            
        except Exception as e:
            logger.error(f"获取最新期号失败: {e}")
            return None
    
    def _spider_500wan_data(self, lottery_name: str, start: int, end: str, source_config: Dict) -> List[Dict]:
        """爬取500彩票网历史数据"""
        try:
            # 构建数据URL
            data_url = f"{source_config['data_url']}?start={start}&end={end}"
            
            logger.info(f"请求URL: {data_url}")
            
            response = self._make_request(data_url)
            if not response:
                logger.warning(f"无法获取数据页面: {data_url}")
                return []
            
            # 设置正确的编码
            response.encoding = source_config.get('encoding', 'gb2312')
            
            if not BS4_AVAILABLE:
                logger.error("需要BeautifulSoup库来解析500彩票网数据")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找数据表格
            tbody = soup.find('tbody', {'id': 'tdata'})
            if not tbody:
                logger.warning("未找到数据表格")
                return []
            
            trs = tbody.find_all('tr')
            logger.info(f"找到{len(trs)}行数据")
            
            data = []
            for tr in trs:
                try:
                    tds = tr.find_all('td')
                    if len(tds) < 8:
                        continue
                    
                    item = {}
                    
                    if lottery_name == 'ssq':  # 双色球
                        item['period'] = tds[0].get_text().strip()
                        red_balls = []
                        for i in range(1, 7):
                            red_balls.append(int(tds[i].get_text().strip()))
                        blue_balls = [int(tds[7].get_text().strip())]
                        
                        item['numbers'] = {
                            'red_balls': red_balls,
                            'blue_balls': blue_balls
                        }
                        
                        # 尝试获取日期（如果有的话）
                        if len(tds) > 8:
                            date_text = tds[8].get_text().strip()
                            item['date'] = date_text if date_text else datetime.now().strftime('%Y-%m-%d')
                        else:
                            item['date'] = datetime.now().strftime('%Y-%m-%d')
                    
                    elif lottery_name == 'dlt':  # 大乐透
                        item['period'] = tds[0].get_text().strip()
                        front_area = []
                        for i in range(1, 6):
                            front_area.append(int(tds[i].get_text().strip()))
                        back_area = []
                        for j in range(6, 8):
                            back_area.append(int(tds[j].get_text().strip()))
                        
                        item['numbers'] = {
                            'front_area': front_area,
                            'back_area': back_area
                        }
                        
                        # 尝试获取日期
                        if len(tds) > 8:
                            date_text = tds[8].get_text().strip()
                            item['date'] = date_text if date_text else datetime.now().strftime('%Y-%m-%d')
                        else:
                            item['date'] = datetime.now().strftime('%Y-%m-%d')
                    
                    if item:
                        data.append(item)
                
                except Exception as e:
                    logger.debug(f"解析行数据失败: {e}")
                    continue
            
            logger.info(f"成功解析{len(data)}期数据")
            return data
            
        except Exception as e:
            logger.error(f"爬取500彩票网数据失败: {e}")
            return []
    
    def _fetch_from_api(self, lottery_code: str, source_config: Dict, periods: int) -> List[Dict]:
        """从API接口获取数据"""
        try:
            url = source_config['url']
            method = source_config.get('method', 'GET')
            params = source_config.get('params', {})
            
            logger.info(f"调用API: {source_config['name']} ({method})")
            
            # 准备请求参数
            if method == 'GET':
                # 对于GET请求，将参数作为URL参数
                response = self._make_request(url, method='GET', params=params)
            else:
                # 对于POST请求，将参数作为数据发送
                response = self._make_request(url, method='POST', data=params)
            
            if not response:
                logger.warning(f"API请求失败: {url}")
                return []
            
            # 尝试解析JSON响应
            try:
                data = response.json()
                logger.info(f"成功获取API数据，解析JSON")
                return self._parse_api_response(data, lottery_code, periods, source_config['name'])
            except json.JSONDecodeError:
                logger.warning(f"API响应不是有效JSON，尝试文本解析")
                return self._parse_text_response(response.text, lottery_code, periods)
                
        except Exception as e:
            logger.error(f"从API获取数据失败 ({source_config.get('name', 'Unknown')}): {e}")
            return []
    
    def _parse_api_response(self, data: Dict, lottery_code: str, periods: int, source_name: str) -> List[Dict]:
        """解析API响应数据"""
        try:
            results = []
            
            if source_name == '中国福彩网官方':
                # 解析福彩网官方格式
                if 'result' in data and isinstance(data['result'], list):
                    for item in data['result'][:periods]:
                        try:
                            period = item.get('code')
                            date = item.get('date')
                            red = item.get('red', '').split(',')
                            blue = item.get('blue', '').split(',')
                            
                            if period and red and blue:
                                result = {
                                    'period': period,
                                    'date': date,
                                    'numbers': {
                                        'red_balls': [int(x) for x in red if x.isdigit()],
                                        'blue_balls': [int(x) for x in blue if x.isdigit()]
                                    }
                                }
                                results.append(result)
                        except Exception as e:
                            logger.debug(f"解析福彩网数据项失败: {e}")
                            continue
            
            elif source_name == '中国体彩网官方':
                # 解析体彩网官方格式
                if 'value' in data and 'list' in data['value']:
                    for item in data['value']['list'][:periods]:
                        try:
                            period = item.get('lotteryDrawNum')
                            date = item.get('lotteryDrawTime')
                            draw_result = item.get('lotteryDrawResult', '')
                            
                            if period and draw_result:
                                # 解析大乐透格式: "01 02 03 04 05|06 07"
                                parts = draw_result.split('|')
                                if len(parts) == 2:
                                    front = [int(x) for x in parts[0].split() if x.isdigit()]
                                    back = [int(x) for x in parts[1].split() if x.isdigit()]
                                    
                                    result = {
                                        'period': period,
                                        'date': date,
                                        'numbers': {
                                            'front_area': front,
                                            'back_area': back
                                        }
                                    }
                                    results.append(result)
                        except Exception as e:
                            logger.debug(f"解析体彩网数据项失败: {e}")
                            continue
            
            elif source_name == '新浪彩票接口':
                # 解析新浪格式
                if 'result' in data and 'data' in data['result']:
                    lottery_data = data['result']['data']
                    if isinstance(lottery_data, list):
                        for item in lottery_data[:periods]:
                            try:
                                period = item.get('expect')
                                date = item.get('opentime')
                                opencode = item.get('opencode')
                                
                                if period and opencode:
                                    numbers = self._parse_opencode_format(opencode, lottery_code)
                                    if numbers:
                                        result = {
                                            'period': period,
                                            'date': date,
                                            'numbers': numbers
                                        }
                                        results.append(result)
                            except Exception as e:
                                logger.debug(f"解析新浪数据项失败: {e}")
                                continue
            
            elif source_name == 'OpenLottery免费接口':
                # 解析OpenLottery格式
                if 'data' in data and isinstance(data['data'], list):
                    for item in data['data'][:periods]:
                        try:
                            period = item.get('issue')
                            date = item.get('date')
                            numbers_data = item.get('number')
                            
                            if period and numbers_data:
                                if lottery_code == 'shuangseqiu':
                                    red = numbers_data.get('red', [])
                                    blue = numbers_data.get('blue', [])
                                    
                                    result = {
                                        'period': period,
                                        'date': date,
                                        'numbers': {
                                            'red_balls': red,
                                            'blue_balls': blue
                                        }
                                    }
                                else:  # 大乐透
                                    front = numbers_data.get('front', [])
                                    back = numbers_data.get('back', [])
                                    
                                    result = {
                                        'period': period,
                                        'date': date,
                                        'numbers': {
                                            'front_area': front,
                                            'back_area': back
                                        }
                                    }
                                
                                results.append(result)
                        except Exception as e:
                            logger.debug(f"解析OpenLottery数据项失败: {e}")
                            continue
            
            logger.info(f"从{source_name}解析到{len(results)}期数据")
            return results
            
        except Exception as e:
            logger.error(f"解析API响应失败 ({source_name}): {e}")
            return []
    
    def _parse_opencode_format(self, opencode: str, lottery_code: str) -> Dict:
        """解析开奖号码格式"""
        try:
            # 处理各种格式的分隔符
            clean_code = opencode.replace('+', ',').replace('|', ',').replace(' ', ',')
            numbers = [int(x) for x in clean_code.split(',') if x.strip().isdigit()]
            
            if lottery_code == 'shuangseqiu':
                if len(numbers) >= 7:
                    return {
                        'red_balls': numbers[:6],
                        'blue_balls': [numbers[6]]
                    }
            else:  # 大乐透
                if len(numbers) >= 7:
                    return {
                        'front_area': numbers[:5],
                        'back_area': numbers[5:7]
                    }
            
            return {}
            
        except Exception as e:
            logger.debug(f"解析opencode失败: {e}")
            return {}
    
    def _parse_text_response(self, text: str, lottery_code: str, periods: int) -> List[Dict]:
        """解析文本响应（备用方案）"""
        try:
            # 尝试从文本中提取有用信息
            results = []
            
            # 查找期号和号码
            if lottery_code == 'shuangseqiu':
                pattern = r'(\d{7})[^0-9]*(\d{2})[^0-9]*(\d{2})[^0-9]*(\d{2})[^0-9]*(\d{2})[^0-9]*(\d{2})[^0-9]*(\d{2})[^0-9]*(\d{2})'
                matches = re.findall(pattern, text)
                
                for match in matches[:periods]:
                    try:
                        period = match[0]
                        red_balls = [int(match[i]) for i in range(1, 7)]
                        blue_balls = [int(match[7])]
                        
                        result = {
                            'period': period,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'numbers': {
                                'red_balls': red_balls,
                                'blue_balls': blue_balls
                            }
                        }
                        results.append(result)
                    except Exception as e:
                        logger.debug(f"解析文本匹配失败: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析文本响应失败: {e}")
            return []
    
    def _fetch_from_500wan_html(self, lottery_code: str, source_config: Dict, periods: int) -> List[Dict]:
        """从500彩票网HTML页面获取数据"""
        try:
            url = source_config['url']
            
            # 添加更好的User-Agent和Headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = self._make_request(url, headers=headers)
            if not response:
                logger.warning(f"无法获取页面: {url}")
                return []
            
            logger.info(f"成功获取页面，响应长度: {len(response.text)}")
            
            # 解析页面数据
            return self._parse_modern_500wan_page(response.text, lottery_code, periods)
                
        except Exception as e:
            logger.error(f"从500彩票网HTML获取数据失败: {e}")
            return []
    
    def _fetch_from_500wan_xml(self, lottery_code: str, source_config: Dict, periods: int) -> List[Dict]:
        """从500彩票网XML接口获取数据"""
        try:
            url = source_config['url']
            response = self._make_request(url)
            
            if not response:
                return []
            
            return self._parse_xml_data(response.text, lottery_code, periods)
                
        except Exception as e:
            logger.error(f"从500彩票网XML获取数据失败: {e}")
            return []
    
    def _fetch_from_official(self, lottery_code: str, periods: int) -> List[Dict]:
        """从官方网站获取数据"""
        try:
            source_config = self.data_sources[lottery_code]['secondary']
            
            if 'api_url' in source_config:
                # 使用API接口
                return self._fetch_from_official_api(lottery_code, periods, source_config)
            else:
                # 解析网页
                return self._fetch_from_official_web(lottery_code, periods, source_config)
                
        except Exception as e:
            logger.error(f"从官方网站获取数据失败: {e}")
            return []
    
    def _fetch_from_official_api(self, lottery_code: str, periods: int, 
                                source_config: Dict) -> List[Dict]:
        """从官方API获取数据"""
        try:
            api_url = source_config['api_url']
            
            # 构建API请求参数
            params = {
                'name': 'ssq' if lottery_code == 'shuangseqiu' else 'dlt',
                'issueCount': periods,
                'issueStart': '',
                'issueEnd': ''
            }
            
            response = self._make_request(api_url, method='POST', data=params)
            if not response:
                return []
            
            try:
                data = response.json()
                return self._parse_official_api_response(data, lottery_code)
            except json.JSONDecodeError:
                logger.error("官方API响应不是有效JSON")
                return []
                
        except Exception as e:
            logger.error(f"官方API请求失败: {e}")
            return []
    
    def _fetch_from_official_web(self, lottery_code: str, periods: int, 
                                source_config: Dict) -> List[Dict]:
        """从官方网页获取数据"""
        try:
            if not BS4_AVAILABLE:
                logger.warning("BeautifulSoup不可用，跳过网页解析")
                return []
            
            url = source_config['url']
            response = self._make_request(url)
            
            if not response:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_official_web_page(soup, lottery_code, periods)
            
        except Exception as e:
            logger.error(f"官方网页解析失败: {e}")
            return []
    
    def _make_request(self, url: str, method: str = 'GET', headers: Dict = None, data: Dict = None, params: Dict = None, **kwargs) -> Optional[requests.Response]:
        """发送HTTP请求"""
        for attempt in range(self.max_retries):
            try:
                # 合并自定义headers
                request_headers = self.session.headers.copy()
                if headers:
                    request_headers.update(headers)
                
                if method.upper() == 'GET':
                    response = self.session.get(
                        url, 
                        timeout=self.timeout, 
                        headers=request_headers, 
                        params=params,
                        **kwargs
                    )
                else:
                    response = self.session.post(
                        url, 
                        timeout=self.timeout, 
                        headers=request_headers, 
                        data=data,
                        params=params,
                        **kwargs
                    )
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # 递增延迟
                continue
        
        return None
    
    def _parse_500wan_html(self, html_content: str, lottery_code: str) -> List[Dict]:
        """解析500彩票网HTML内容"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # 查找数据行
            rows = soup.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:  # 确保有足够的列
                    try:
                        period = cells[0].get_text(strip=True)
                        date = cells[1].get_text(strip=True)
                        
                        if lottery_code == 'shuangseqiu':
                            # 双色球：6个红球 + 1个蓝球
                            red_balls = []
                            for i in range(2, 8):
                                red_balls.append(int(cells[i].get_text(strip=True)))
                            blue_balls = [int(cells[8].get_text(strip=True))]
                            
                            result = {
                                'period': period,
                                'date': date,
                                'numbers': {
                                    'red_balls': red_balls,
                                    'blue_balls': blue_balls
                                }
                            }
                        else:  # 大乐透
                            # 大乐透：5个前区 + 2个后区
                            front_area = []
                            for i in range(2, 7):
                                front_area.append(int(cells[i].get_text(strip=True)))
                            back_area = []
                            for i in range(7, 9):
                                back_area.append(int(cells[i].get_text(strip=True)))
                            
                            result = {
                                'period': period,
                                'date': date,
                                'numbers': {
                                    'front_area': front_area,
                                    'back_area': back_area
                                }
                            }
                        
                        results.append(result)
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"解析行数据失败: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析500彩票网HTML失败: {e}")
            return []
    
    def _parse_500wan_text(self, content: str, lottery_code: str) -> List[Dict]:
        """解析500彩票网文本内容（无BeautifulSoup）"""
        try:
            results = []
            
            # 使用正则表达式提取数据
            if lottery_code == 'shuangseqiu':
                # 双色球模式
                pattern = r'(\d{7})\s+(\d{4}-\d{2}-\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})'
            else:
                # 大乐透模式
                pattern = r'(\d{7})\s+(\d{4}-\d{2}-\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})'
            
            matches = re.findall(pattern, content)
            
            for match in matches:
                try:
                    period = match[0]
                    date = match[1]
                    
                    if lottery_code == 'shuangseqiu':
                        red_balls = [int(x) for x in match[2:8]]
                        blue_balls = [int(match[8])]
                        
                        result = {
                            'period': period,
                            'date': date,
                            'numbers': {
                                'red_balls': red_balls,
                                'blue_balls': blue_balls
                            }
                        }
                    else:  # 大乐透
                        front_area = [int(x) for x in match[2:7]]
                        back_area = [int(x) for x in match[7:9]]
                        
                        result = {
                            'period': period,
                            'date': date,
                            'numbers': {
                                'front_area': front_area,
                                'back_area': back_area
                            }
                        }
                    
                    results.append(result)
                    
                except ValueError as e:
                    logger.debug(f"解析数据失败: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析文本内容失败: {e}")
            return []
    
    def _parse_official_api_response(self, data: Dict, lottery_code: str) -> List[Dict]:
        """解析官方API响应"""
        try:
            results = []
            
            if 'result' in data and isinstance(data['result'], list):
                for item in data['result']:
                    try:
                        period = item.get('lotteryDrawNum', '')
                        date = item.get('lotteryDrawTime', '')
                        
                        # 解析开奖号码
                        draw_result = item.get('lotteryDrawResult', '')
                        if draw_result:
                            numbers = self._parse_number_string(draw_result, lottery_code)
                            
                            result = {
                                'period': period,
                                'date': date,
                                'numbers': numbers
                            }
                            results.append(result)
                            
                    except Exception as e:
                        logger.debug(f"解析API数据项失败: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析官方API响应失败: {e}")
            return []
    
    def _parse_official_web_page(self, soup: BeautifulSoup, lottery_code: str, 
                                periods: int) -> List[Dict]:
        """解析官方网页"""
        try:
            results = []
            
            # 这里需要根据具体的官方网站结构来实现
            # 由于官方网站结构可能经常变化，这里提供基础框架
            
            # 查找包含开奖信息的元素
            info_elements = soup.find_all('div', class_='kj_list') or soup.find_all('tr')
            
            for element in info_elements[:periods]:
                try:
                    # 提取期号、日期和号码
                    text = element.get_text()
                    
                    # 使用正则表达式提取信息
                    period_match = re.search(r'(\d{7})', text)
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
                    
                    if period_match and date_match:
                        period = period_match.group(1)
                        date = date_match.group(1)
                        
                        # 提取号码（需要根据具体格式调整）
                        numbers = self._extract_numbers_from_text(text, lottery_code)
                        
                        if numbers:
                            result = {
                                'period': period,
                                'date': date,
                                'numbers': numbers
                            }
                            results.append(result)
                            
                except Exception as e:
                    logger.debug(f"解析网页元素失败: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析官方网页失败: {e}")
            return []
    
    def _parse_number_string(self, number_string: str, lottery_code: str) -> Dict:
        """解析号码字符串"""
        try:
            # 移除空格和特殊字符
            clean_string = re.sub(r'[^\d\s+]', ' ', number_string)
            numbers = [int(x) for x in clean_string.split() if x.isdigit()]
            
            if lottery_code == 'shuangseqiu':
                # 双色球：前6个是红球，最后1个是蓝球
                if len(numbers) >= 7:
                    return {
                        'red_balls': numbers[:6],
                        'blue_balls': [numbers[6]]
                    }
            else:  # 大乐透
                # 大乐透：前5个是前区，后2个是后区
                if len(numbers) >= 7:
                    return {
                        'front_area': numbers[:5],
                        'back_area': numbers[5:7]
                    }
            
            return {}
            
        except Exception as e:
            logger.debug(f"解析号码字符串失败: {e}")
            return {}
    
    def _extract_numbers_from_text(self, text: str, lottery_code: str) -> Dict:
        """从文本中提取号码"""
        try:
            # 提取所有数字
            numbers = re.findall(r'\d+', text)
            numbers = [int(x) for x in numbers if 1 <= int(x) <= 35]  # 过滤有效范围
            
            if lottery_code == 'shuangseqiu' and len(numbers) >= 7:
                return {
                    'red_balls': numbers[:6],
                    'blue_balls': [numbers[6]]
                }
            elif lottery_code == 'daletou' and len(numbers) >= 7:
                return {
                    'front_area': numbers[:5],
                    'back_area': numbers[5:7]
                }
            
            return {}
            
        except Exception as e:
            logger.debug(f"从文本提取号码失败: {e}")
            return {}
    
    def _fetch_from_sina(self, lottery_code: str, periods: int) -> List[Dict]:
        """从新浪彩票获取数据"""
        try:
            source_config = self.data_sources[lottery_code]['tertiary']
            api_url = source_config['api_url']
            
            # 修正URL中的彩票类型参数
            if lottery_code == 'shuangseqiu':
                api_url = api_url.replace('lotid=dlt', 'lotid=ssq')
            
            # 构建请求参数
            params = {
                'dpc': 1,
                'lotid': 'ssq' if lottery_code == 'shuangseqiu' else 'dlt',
                'num': min(periods, 200)
            }
            
            response = self._make_request(api_url, params=params)
            if not response:
                return []
            
            try:
                data = response.json()
                return self._parse_sina_response(data, lottery_code)
            except json.JSONDecodeError:
                logger.error("新浪API响应不是有效JSON")
                return []
                
        except Exception as e:
            logger.error(f"从新浪彩票获取数据失败: {e}")
            return []
    
    def _fetch_from_netease(self, lottery_code: str, periods: int) -> List[Dict]:
        """从网易彩票获取数据"""
        try:
            source_config = self.data_sources[lottery_code]['fallback']
            api_url = source_config['api_url']
            
            # 构建请求参数
            params = {
                'lotid': '01001' if lottery_code == 'shuangseqiu' else '01007',
                'type': 'history',
                'pagesize': min(periods, 200)
            }
            
            response = self._make_request(api_url, params=params)
            if not response:
                return []
            
            try:
                data = response.json()
                return self._parse_netease_response(data, lottery_code)
            except json.JSONDecodeError:
                logger.error("网易API响应不是有效JSON")
                return []
                
        except Exception as e:
            logger.error(f"从网易彩票获取数据失败: {e}")
            return []
    
    def _parse_sina_response(self, data: Dict, lottery_code: str) -> List[Dict]:
        """解析新浪API响应"""
        try:
            results = []
            
            if 'result' in data and 'data' in data['result']:
                for item in data['result']['data']:
                    try:
                        period = item.get('expect', '')
                        date = item.get('opentime', '')
                        opencode = item.get('opencode', '')
                        
                        if opencode:
                            numbers = self._parse_opencode(opencode, lottery_code)
                            if numbers:
                                result = {
                                    'period': period,
                                    'date': date,
                                    'numbers': numbers
                                }
                                results.append(result)
                                
                    except Exception as e:
                        logger.debug(f"解析新浪数据项失败: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析新浪API响应失败: {e}")
            return []
    
    def _parse_netease_response(self, data: Dict, lottery_code: str) -> List[Dict]:
        """解析网易API响应"""
        try:
            results = []
            
            if 'data' in data and isinstance(data['data'], list):
                for item in data['data']:
                    try:
                        period = item.get('period', '')
                        date = item.get('date', '')
                        numbers = item.get('number', '')
                        
                        if numbers:
                            parsed_numbers = self._parse_number_string(numbers, lottery_code)
                            if parsed_numbers:
                                result = {
                                    'period': period,
                                    'date': date,
                                    'numbers': parsed_numbers
                                }
                                results.append(result)
                                
                    except Exception as e:
                        logger.debug(f"解析网易数据项失败: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析网易API响应失败: {e}")
            return []
    
    def _parse_opencode(self, opencode: str, lottery_code: str) -> Dict:
        """解析开奖号码字符串"""
        try:
            # 移除特殊字符，只保留数字和逗号
            clean_code = re.sub(r'[^\d,]', '', opencode)
            numbers = [int(x) for x in clean_code.split(',') if x.isdigit()]
            
            if lottery_code == 'shuangseqiu':
                # 双色球：前6个是红球，最后1个是蓝球
                if len(numbers) >= 7:
                    return {
                        'red_balls': numbers[:6],
                        'blue_balls': [numbers[6]]
                    }
            else:  # 大乐透
                # 大乐透：前5个是前区，后2个是后区
                if len(numbers) >= 7:
                    return {
                        'front_area': numbers[:5],
                        'back_area': numbers[5:7]
                    }
            
            return {}
            
        except Exception as e:
            logger.debug(f"解析开奖号码失败: {e}")
            return {}
    
    def _get_current_period(self, lottery_code: str, date: datetime) -> str:
        """获取当前期号"""
        try:
            year = date.year
            
            if lottery_code == 'shuangseqiu':
                # 双色球每周二、四、日开奖
                # 简化计算：假设每年有150期左右
                day_of_year = date.timetuple().tm_yday
                estimated_period = min(150, (day_of_year // 2) + 1)
                return f"{year}{estimated_period:03d}"
            else:  # 大乐透
                # 大乐透每周一、三、六开奖
                # 简化计算：假设每年有150期左右
                day_of_year = date.timetuple().tm_yday
                estimated_period = min(150, (day_of_year // 2) + 1)
                return f"{year}{estimated_period:03d}"
                
        except Exception:
            # 默认返回
            return f"{datetime.now().year}001"
    
    def _calculate_start_period(self, lottery_code: str, end_period: str, 
                              periods: int) -> str:
        """计算开始期号"""
        try:
            year = int(end_period[:4])
            period_num = int(end_period[4:])
            
            start_period_num = max(1, period_num - periods + 1)
            
            return f"{year}{start_period_num:03d}"
            
        except Exception:
            return end_period
    
    def _fetch_historical_range(self, lottery_code: str, start_period: str, 
                              end_period: str) -> List[Dict]:
        """获取历史数据范围"""
        try:
            # 计算需要获取的期数
            start_num = int(start_period[4:])
            end_num = int(end_period[4:])
            periods = end_num - start_num + 1
            
            if periods <= 0:
                return []
            
            # 使用现有方法获取数据
            try:
                source_config = self.data_sources[lottery_code]['primary']
                return self._fetch_from_500wan_correct(lottery_code, source_config, min(periods, 100))
            except Exception as e:
                logger.error(f"从500彩票网获取历史范围数据失败: {e}")
                return []
            
        except Exception as e:
            logger.error(f"获取历史范围数据失败: {e}")
            return []
    
    def _update_cache(self, lottery_type: str, data: List[Dict]):
        """更新缓存"""
        with self.lock:
            self.latest_data_cache[lottery_type] = data
            self.cache_timestamp[lottery_type] = datetime.now()
    
    def _get_cached_data(self, lottery_type: str, periods: int) -> List[Dict]:
        """获取缓存数据"""
        with self.lock:
            cached_data = self.latest_data_cache.get(lottery_type, [])
            return cached_data[:periods] if cached_data else []
    
    def is_cache_valid(self, lottery_type: str, max_age_hours: int = 24) -> bool:
        """检查缓存是否有效"""
        with self.lock:
            if lottery_type not in self.cache_timestamp:
                return False
            
            cache_time = self.cache_timestamp[lottery_type]
            age = datetime.now() - cache_time
            
            return age.total_seconds() < max_age_hours * 3600
    
    def clear_cache(self, lottery_type: Optional[str] = None):
        """清除缓存"""
        with self.lock:
            if lottery_type:
                self.latest_data_cache.pop(lottery_type, None)
                self.cache_timestamp.pop(lottery_type, None)
            else:
                self.latest_data_cache.clear()
                self.cache_timestamp.clear()
    
    def get_cache_status(self) -> Dict:
        """获取缓存状态"""
        with self.lock:
            status = {}
            for lottery_type in self.latest_data_cache:
                cache_time = self.cache_timestamp.get(lottery_type)
                data_count = len(self.latest_data_cache[lottery_type])
                
                status[lottery_type] = {
                    'data_count': data_count,
                    'cache_time': cache_time.isoformat() if cache_time else None,
                    'age_hours': ((datetime.now() - cache_time).total_seconds() / 3600) if cache_time else None
                }
            
            return status
    
    def get_all_historical_data(self, lottery_type: str, max_periods: int = 2000) -> List[Dict]:
        """
        获取全部历史开奖数据
        
        Args:
            lottery_type: 彩票类型 ('双色球' 或 '大乐透')
            max_periods: 最大获取期数
            
        Returns:
            全部历史数据列表
        """
        try:
            logger.info(f"开始获取{lottery_type}全部历史数据，最多{max_periods}期...")
            
            lottery_code = 'shuangseqiu' if lottery_type == '双色球' else 'daletou'
            all_data = []
            
            # 分批获取数据，避免单次请求过大
            batch_size = 200
            total_fetched = 0
            
            while total_fetched < max_periods:
                current_batch_size = min(batch_size, max_periods - total_fetched)
                
                # 尝试所有数据源
                batch_data = []
                for source_type in ['primary', 'secondary', 'tertiary', 'fallback']:
                    try:
                        logger.info(f"尝试从{source_type}源获取第{total_fetched+1}-{total_fetched+current_batch_size}期数据...")
                        batch_data = self._fetch_batch_data(lottery_code, source_type, current_batch_size, total_fetched)
                        
                        if batch_data:
                            logger.info(f"成功从{source_type}源获取{len(batch_data)}期数据")
                            break
                    except Exception as e:
                        logger.warning(f"从{source_type}源获取批量数据失败: {e}")
                        continue
                
                if not batch_data:
                    logger.warning(f"所有数据源都无法获取第{total_fetched+1}批数据，停止获取")
                    break
                
                # 去重并添加到总数据中
                for item in batch_data:
                    if not any(existing['period'] == item['period'] for existing in all_data):
                        all_data.append(item)
                
                total_fetched += len(batch_data)
                
                # 如果获取的数据少于预期，可能已经到达数据源的最大范围
                if len(batch_data) < current_batch_size:
                    logger.info(f"数据源返回数据少于预期，可能已获取全部可用数据")
                    break
                
                # 短暂延迟，避免频繁请求
                time.sleep(0.5)
            
            # 按期号排序
            all_data.sort(key=lambda x: x.get('period', ''), reverse=True)
            
            # 更新缓存
            if all_data:
                self._update_cache(lottery_type, all_data)
            
            logger.info(f"完成{lottery_type}历史数据获取，共获得{len(all_data)}期数据")
            return all_data
            
        except Exception as e:
            logger.error(f"获取全部历史数据失败: {e}")
            return []
    
    def _fetch_batch_data(self, lottery_code: str, source_type: str, 
                         batch_size: int, offset: int = 0) -> List[Dict]:
        """获取批量数据"""
        try:
            source_config = self.data_sources[lottery_code][source_type]
            
            if source_type == 'primary':
                return self._fetch_batch_from_500wan(lottery_code, batch_size, offset)
            elif source_type == 'secondary':
                return self._fetch_batch_from_official(lottery_code, batch_size, offset)
            elif source_type == 'tertiary':
                return self._fetch_batch_from_sina(lottery_code, batch_size, offset)
            elif source_type == 'fallback':
                return self._fetch_batch_from_netease(lottery_code, batch_size, offset)
            
            return []
            
        except Exception as e:
            logger.error(f"获取批量数据失败: {e}")
            return []
    
    def _fetch_batch_from_500wan(self, lottery_code: str, batch_size: int, offset: int) -> List[Dict]:
        """从500彩票网获取批量数据"""
        try:
            source_config = self.data_sources[lottery_code]['primary']
            return self._fetch_from_500wan_correct(lottery_code, source_config, batch_size)
        except Exception as e:
            logger.error(f"从500彩票网获取批量数据失败: {e}")
            return []
    
    def _fetch_batch_from_official(self, lottery_code: str, batch_size: int, offset: int) -> List[Dict]:
        """从官方网站获取批量数据"""
        try:
            source_config = self.data_sources[lottery_code]['secondary']
            return self._fetch_from_api(lottery_code, source_config, batch_size)
        except Exception as e:
            logger.error(f"从官方网站获取批量数据失败: {e}")
            return []
    
    def _fetch_batch_from_sina(self, lottery_code: str, batch_size: int, offset: int) -> List[Dict]:
        """从新浪彩票获取批量数据"""
        try:
            source_config = self.data_sources[lottery_code]['tertiary']
            return self._fetch_from_api(lottery_code, source_config, batch_size)
        except Exception as e:
            logger.error(f"从新浪彩票获取批量数据失败: {e}")
            return []
    
    def _fetch_batch_from_netease(self, lottery_code: str, batch_size: int, offset: int) -> List[Dict]:
        """从网易彩票获取批量数据"""
        try:
            source_config = self.data_sources[lottery_code]['fallback']
            return self._fetch_from_local_cache(lottery_code, batch_size)
        except Exception as e:
            logger.error(f"从缓存获取批量数据失败: {e}")
            return []
    
    def auto_update_data(self, lottery_type: str, check_interval_hours: int = 2) -> bool:
        """
        自动更新数据
        
        Args:
            lottery_type: 彩票类型
            check_interval_hours: 检查间隔小时数
            
        Returns:
            是否有新数据更新
        """
        try:
            # 检查缓存是否需要更新
            if self.is_cache_valid(lottery_type, check_interval_hours):
                logger.debug(f"{lottery_type}缓存仍然有效，无需更新")
                return False
            
            logger.info(f"开始自动更新{lottery_type}数据...")
            
            # 获取最新数据
            latest_data = self.get_latest_data(lottery_type, 10)
            
            if latest_data:
                logger.info(f"自动更新成功，获得{len(latest_data)}期最新数据")
                return True
            else:
                logger.warning(f"自动更新失败，未获得新数据")
                return False
                
        except Exception as e:
            logger.error(f"自动更新数据失败: {e}")
            return False
    

    
    def _fetch_from_local_cache(self, lottery_code: str, periods: int) -> List[Dict]:
        """从本地缓存获取数据"""
        try:
            lottery_type = '双色球' if lottery_code == 'shuangseqiu' else '大乐透'
            
            # 检查是否有缓存数据
            cached_data = self._get_cached_data(lottery_type, periods)
            
            if cached_data:
                logger.info(f"从本地缓存获取{len(cached_data)}期{lottery_type}数据")
                return cached_data
            
            # 尝试从历史JSON文件读取
            cache_file_path = os.path.join('history_data', 'history_cache.json')
            if os.path.exists(cache_file_path):
                try:
                    with open(cache_file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if lottery_type in cache_data:
                        history_data = cache_data[lottery_type]
                        if isinstance(history_data, list) and len(history_data) > 0:
                            # 转换格式
                            results = []
                            for item in history_data[:periods]:
                                if isinstance(item, dict) and 'period' in item:
                                    results.append(item)
                            
                            if results:
                                logger.info(f"从历史缓存文件获取{len(results)}期{lottery_type}数据")
                                return results
                except Exception as e:
                    logger.warning(f"读取历史缓存文件失败: {e}")
            
            logger.warning(f"本地缓存中没有{lottery_type}数据")
            return []
            
        except Exception as e:
            logger.error(f"从本地缓存获取数据失败: {e}")
            return []
    

    
    def _parse_history_page(self, html_content: str, lottery_code: str, periods: int) -> List[Dict]:
        """解析500彩票网历史数据页面"""
        try:
            results = []
            
            # 尝试多种解析策略
            if BS4_AVAILABLE:
                results = self._parse_with_beautifulsoup(html_content, lottery_code, periods)
            
            if not results:
                results = self._parse_with_regex(html_content, lottery_code, periods)
            
            return results[:periods] if results else []
            
        except Exception as e:
            logger.error(f"解析历史页面失败: {e}")
            return []
    
    def _parse_with_beautifulsoup(self, html_content: str, lottery_code: str, periods: int) -> List[Dict]:
        """使用BeautifulSoup解析HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # 查找包含开奖数据的表格
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows[1:]:  # 跳过表头
                    cells = row.find_all(['td', 'th'])
                    
                    if len(cells) >= 8:  # 确保有足够的数据列
                        try:
                            period = cells[0].get_text(strip=True)
                            date = cells[1].get_text(strip=True)
                            
                            # 验证期号格式
                            if not re.match(r'\d{7}', period):
                                continue
                            
                            if lottery_code == 'shuangseqiu':
                                # 双色球：6个红球 + 1个蓝球
                                red_balls = []
                                for i in range(2, 8):
                                    ball_text = cells[i].get_text(strip=True)
                                    if ball_text.isdigit():
                                        red_balls.append(int(ball_text))
                                
                                blue_text = cells[8].get_text(strip=True) if len(cells) > 8 else ''
                                if blue_text.isdigit():
                                    blue_balls = [int(blue_text)]
                                else:
                                    continue
                                
                                if len(red_balls) == 6:
                                    result = {
                                        'period': period,
                                        'date': date,
                                        'numbers': {
                                            'red_balls': red_balls,
                                            'blue_balls': blue_balls
                                        }
                                    }
                                    results.append(result)
                            
                            else:  # 大乐透
                                # 大乐透：5个前区 + 2个后区
                                front_area = []
                                for i in range(2, 7):
                                    ball_text = cells[i].get_text(strip=True)
                                    if ball_text.isdigit():
                                        front_area.append(int(ball_text))
                                
                                back_area = []
                                for i in range(7, 9):
                                    if i < len(cells):
                                        ball_text = cells[i].get_text(strip=True)
                                        if ball_text.isdigit():
                                            back_area.append(int(ball_text))
                                
                                if len(front_area) == 5 and len(back_area) == 2:
                                    result = {
                                        'period': period,
                                        'date': date,
                                        'numbers': {
                                            'front_area': front_area,
                                            'back_area': back_area
                                        }
                                    }
                                    results.append(result)
                            
                        except (ValueError, IndexError) as e:
                            logger.debug(f"解析行数据失败: {e}")
                            continue
                
                if results:  # 如果找到数据就停止搜索其他表格
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"BeautifulSoup解析失败: {e}")
            return []
    
    def _parse_with_regex(self, html_content: str, lottery_code: str, periods: int) -> List[Dict]:
        """使用正则表达式解析HTML"""
        try:
            results = []
            
            if lottery_code == 'shuangseqiu':
                # 双色球正则模式 - 匹配期号、日期和号码
                pattern = r'(\d{7})[^>]*>([^<]*\d{4}-\d{2}-\d{2}[^<]*)</[^>]*>[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})'
                matches = re.findall(pattern, html_content)
                
                for match in matches:
                    try:
                        period = match[0]
                        date = re.search(r'\d{4}-\d{2}-\d{2}', match[1])
                        date = date.group() if date else match[1]
                        
                        red_balls = [int(match[i]) for i in range(2, 8)]
                        blue_balls = [int(match[8])]
                        
                        result = {
                            'period': period,
                            'date': date,
                            'numbers': {
                                'red_balls': red_balls,
                                'blue_balls': blue_balls
                            }
                        }
                        results.append(result)
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"正则解析行失败: {e}")
                        continue
            
            else:  # 大乐透
                # 大乐透正则模式
                pattern = r'(\d{7})[^>]*>([^<]*\d{4}-\d{2}-\d{2}[^<]*)</[^>]*>[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})[^>]*>(\d{2})'
                matches = re.findall(pattern, html_content)
                
                for match in matches:
                    try:
                        period = match[0]
                        date = re.search(r'\d{4}-\d{2}-\d{2}', match[1])
                        date = date.group() if date else match[1]
                        
                        front_area = [int(match[i]) for i in range(2, 7)]
                        back_area = [int(match[i]) for i in range(7, 9)]
                        
                        result = {
                            'period': period,
                            'date': date,
                            'numbers': {
                                'front_area': front_area,
                                'back_area': back_area
                            }
                        }
                        results.append(result)
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"正则解析行失败: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"正则表达式解析失败: {e}")
            return []
    
    def _estimate_current_period_number(self, lottery_code: str) -> int:
        """估算当前期号"""
        try:
            now = datetime.now()
            day_of_year = now.timetuple().tm_yday
            
            if lottery_code == 'shuangseqiu':
                # 双色球每周二、四、日开奖，大约每年150期
                estimated = min(150, max(1, (day_of_year * 150) // 365))
            else:  # 大乐透
                # 大乐透每周一、三、六开奖，大约每年150期
                estimated = min(150, max(1, (day_of_year * 150) // 365))
            
            return estimated
            
        except Exception:
            return 100  # 默认值
    
    def _parse_modern_500wan_page(self, html_content: str, lottery_code: str, periods: int) -> List[Dict]:
        """解析现代500彩票网页面"""
        try:
            results = []
            
            # 先尝试从页面脚本中提取JSON数据
            json_data = self._extract_json_from_page(html_content)
            if json_data:
                results = self._parse_json_data(json_data, lottery_code, periods)
                if results:
                    return results
            
            # 如果JSON提取失败，尝试HTML解析
            if BS4_AVAILABLE:
                results = self._parse_modern_html_with_soup(html_content, lottery_code, periods)
            
            if not results:
                results = self._parse_modern_html_with_regex(html_content, lottery_code, periods)
            
            return results[:periods] if results else []
            
        except Exception as e:
            logger.error(f"解析现代500彩票网页面失败: {e}")
            return []
    
    def _extract_json_from_page(self, html_content: str) -> Dict:
        """从页面中提取JSON数据"""
        try:
            # 查找页面中的JSON数据
            # 常见模式：var data = {...}; 或 window.data = {...};
            patterns = [
                r'var\s+\w*data\w*\s*=\s*(\{.*?\});',
                r'window\.\w*data\w*\s*=\s*(\{.*?\});',
                r'lottery_data\s*=\s*(\{.*?\});',
                r'history_data\s*=\s*(\[.*?\]);'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        # 清理JSON字符串
                        clean_json = match.strip()
                        if clean_json.endswith(';'):
                            clean_json = clean_json[:-1]
                        
                        data = json.loads(clean_json)
                        if data:
                            logger.info("成功从页面提取JSON数据")
                            return data
                    except json.JSONDecodeError:
                        continue
            
            return {}
            
        except Exception as e:
            logger.debug(f"从页面提取JSON失败: {e}")
            return {}
    
    def _parse_json_data(self, data: Dict, lottery_code: str, periods: int) -> List[Dict]:
        """解析从页面提取的JSON数据"""
        try:
            results = []
            
            # 根据不同的JSON结构进行解析
            if isinstance(data, list):
                # 如果直接是数组
                data_list = data
            elif 'list' in data:
                # 如果数据在list字段中
                data_list = data['list']
            elif 'data' in data:
                # 如果数据在data字段中
                if isinstance(data['data'], list):
                    data_list = data['data']
                elif 'list' in data['data']:
                    data_list = data['data']['list']
                else:
                    data_list = []
            else:
                data_list = []
            
            for item in data_list[:periods]:
                try:
                    if isinstance(item, dict):
                        period = item.get('expect') or item.get('period') or item.get('qihao')
                        date = item.get('date') or item.get('time') or item.get('opentime')
                        numbers = item.get('opencode') or item.get('numbers') or item.get('code')
                        
                        if period and numbers:
                            parsed_numbers = self._parse_number_string_v2(numbers, lottery_code)
                            if parsed_numbers:
                                result = {
                                    'period': str(period),
                                    'date': str(date) if date else datetime.now().strftime('%Y-%m-%d'),
                                    'numbers': parsed_numbers
                                }
                                results.append(result)
                                
                except Exception as e:
                    logger.debug(f"解析JSON数据项失败: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析JSON数据失败: {e}")
            return []
    
    def _parse_modern_html_with_soup(self, html_content: str, lottery_code: str, periods: int) -> List[Dict]:
        """使用BeautifulSoup解析现代HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # 查找可能包含开奖数据的各种元素
            selectors = [
                '.kj-list tr',  # 开奖列表行
                '.history-list tr',  # 历史记录行
                '.result-item',  # 结果项
                '[class*="ball"]',  # 包含ball的class
                '.lottery-result',  # 彩票结果
                'tr[data-period]',  # 带期号属性的行
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    logger.info(f"找到{len(elements)}个数据元素 (选择器: {selector})")
                    
                    for element in elements[:periods * 2]:  # 多取一些以防格式问题
                        try:
                            period, date, numbers = self._extract_data_from_element(element, lottery_code)
                            
                            if period and numbers:
                                result = {
                                    'period': period,
                                    'date': date or datetime.now().strftime('%Y-%m-%d'),
                                    'numbers': numbers
                                }
                                
                                # 避免重复
                                if not any(r['period'] == period for r in results):
                                    results.append(result)
                                    
                        except Exception as e:
                            logger.debug(f"解析元素失败: {e}")
                            continue
                    
                    if results:
                        break
            
            return results[:periods]
            
        except Exception as e:
            logger.error(f"现代HTML解析失败: {e}")
            return []
    
    def _extract_data_from_element(self, element, lottery_code: str):
        """从HTML元素中提取数据"""
        try:
            # 提取期号
            period = None
            period_elem = element.find(class_=re.compile(r'period|qihao|expect'))
            if period_elem:
                period = period_elem.get_text(strip=True)
            
            # 如果没找到，尝试从属性中获取
            if not period:
                period = element.get('data-period') or element.get('data-expect')
            
            # 如果还没找到，从文本中匹配
            if not period:
                text = element.get_text()
                period_match = re.search(r'(\d{7})', text)
                if period_match:
                    period = period_match.group(1)
            
            # 提取日期
            date = None
            date_elem = element.find(class_=re.compile(r'date|time'))
            if date_elem:
                date = date_elem.get_text(strip=True)
            
            # 如果没找到日期，从文本中匹配
            if not date:
                text = element.get_text()
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
                if date_match:
                    date = date_match.group(1)
            
            # 提取号码
            numbers = None
            
            # 查找球号元素
            ball_elements = element.find_all(class_=re.compile(r'ball|num'))
            if ball_elements:
                ball_numbers = []
                for ball_elem in ball_elements:
                    ball_text = ball_elem.get_text(strip=True)
                    if ball_text.isdigit():
                        ball_numbers.append(int(ball_text))
                
                if ball_numbers:
                    numbers = self._format_ball_numbers(ball_numbers, lottery_code)
            
            # 如果没找到球号，尝试从整个元素文本中提取
            if not numbers:
                all_numbers = re.findall(r'\b(\d{2})\b', element.get_text())
                if len(all_numbers) >= 6:
                    ball_numbers = [int(x) for x in all_numbers[:9]]  # 最多取9个数字
                    numbers = self._format_ball_numbers(ball_numbers, lottery_code)
            
            return period, date, numbers
            
        except Exception as e:
            logger.debug(f"提取元素数据失败: {e}")
            return None, None, None
    
    def _format_ball_numbers(self, ball_numbers: List[int], lottery_code: str) -> Dict:
        """格式化球号数据"""
        try:
            if lottery_code == 'shuangseqiu':
                # 双色球：前6个红球，最后1个蓝球
                if len(ball_numbers) >= 7:
                    return {
                        'red_balls': ball_numbers[:6],
                        'blue_balls': [ball_numbers[6]]
                    }
            else:  # 大乐透
                # 大乐透：前5个前区，后2个后区
                if len(ball_numbers) >= 7:
                    return {
                        'front_area': ball_numbers[:5],
                        'back_area': ball_numbers[5:7]
                    }
            
            return {}
            
        except Exception as e:
            logger.debug(f"格式化球号失败: {e}")
            return {}
    
    def _parse_number_string_v2(self, number_string: str, lottery_code: str) -> Dict:
        """解析号码字符串（改进版）"""
        try:
            # 处理各种分隔符和格式
            clean_string = str(number_string)
            
            # 移除HTML标签
            clean_string = re.sub(r'<[^>]+>', ' ', clean_string)
            
            # 替换各种分隔符
            clean_string = re.sub(r'[+|丨,，\s]+', ',', clean_string)
            
            # 提取所有数字
            numbers = re.findall(r'\d+', clean_string)
            numbers = [int(x) for x in numbers if x.isdigit() and 1 <= int(x) <= 35]
            
            if lottery_code == 'shuangseqiu':
                if len(numbers) >= 7:
                    return {
                        'red_balls': numbers[:6],
                        'blue_balls': [numbers[6]]
                    }
            else:  # 大乐透
                if len(numbers) >= 7:
                    return {
                        'front_area': numbers[:5],
                        'back_area': numbers[5:7]
                    }
            
            return {}
            
        except Exception as e:
            logger.debug(f"解析号码字符串失败: {e}")
            return {}
    
    def _parse_modern_html_with_regex(self, html_content: str, lottery_code: str, periods: int) -> List[Dict]:
        """使用正则表达式解析现代HTML"""
        try:
            results = []
            
            # 更灵活的正则模式
            if lottery_code == 'shuangseqiu':
                # 双色球模式 - 匹配各种可能的格式
                patterns = [
                    r'期号[：:]\s*(\d{7})[^>]*开奖时间[：:]\s*([^<]*\d{4}-\d{2}-\d{2}[^<]*)[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})',
                    r'(\d{7})[^>]*?([^<]*\d{4}-\d{2}-\d{2}[^<]*)[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})',
                ]
            else:
                # 大乐透模式
                patterns = [
                    r'期号[：:]\s*(\d{7})[^>]*开奖时间[：:]\s*([^<]*\d{4}-\d{2}-\d{2}[^<]*)[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})',
                    r'(\d{7})[^>]*?([^<]*\d{4}-\d{2}-\d{2}[^<]*)[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})[^>]*?(\d{2})',
                ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                
                for match in matches[:periods]:
                    try:
                        period = match[0]
                        date = re.search(r'\d{4}-\d{2}-\d{2}', match[1])
                        date = date.group() if date else match[1]
                        
                        if lottery_code == 'shuangseqiu':
                            red_balls = [int(match[i]) for i in range(2, 8)]
                            blue_balls = [int(match[8])]
                            
                            result = {
                                'period': period,
                                'date': date,
                                'numbers': {
                                    'red_balls': red_balls,
                                    'blue_balls': blue_balls
                                }
                            }
                        else:  # 大乐透
                            front_area = [int(match[i]) for i in range(2, 7)]
                            back_area = [int(match[i]) for i in range(7, 9)]
                            
                            result = {
                                'period': period,
                                'date': date,
                                'numbers': {
                                    'front_area': front_area,
                                    'back_area': back_area
                                }
                            }
                        
                        results.append(result)
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"正则解析行失败: {e}")
                        continue
                
                if results:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"现代正则表达式解析失败: {e}")
            return []
